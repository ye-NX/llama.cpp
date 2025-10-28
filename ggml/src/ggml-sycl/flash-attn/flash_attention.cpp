// flash_attention.cpp
#include "flash_attention.hpp"
#include "kernels/flash_kernel.hpp"

// Legacy function - kept for backward compatibility
void flash_attention_forward(
    sycl::queue& q,
    float* Q, float* K, float* V, float* O,
    int N, int d
) {
    // Buffer allocation
    sycl::buffer<float, 2> Q_buf(Q, sycl::range<2>(N, d));
    sycl::buffer<float, 2> K_buf(K, sycl::range<2>(N, d));
    sycl::buffer<float, 2> V_buf(V, sycl::range<2>(N, d));
    sycl::buffer<float, 2> O_buf(O, sycl::range<2>(N, d));

    run_flash_attention_kernel(q, Q_buf, K_buf, V_buf, O_buf, N, d);
}

// Main Flash Attention implementation
void flash_attention_sycl(sycl::queue &q,
                          const float *Q,
                          const float *K,
                          const float *V,
                          float *O,
                          const int N,
                          const int d,
                          const size_t M) {
    // Block sizes (from Flash Attention algorithm)
    int Bc = M / (4 * d);
    int Br = std::min(Bc, d);
    int Tr = (N + Br - 1) / Br;  // number of Q blocks
    int Tc = (N + Bc - 1) / Bc;  // number of K/V blocks
    
    // Initialize O to zero as per Flash Attention algorithm step 2
    std::fill(O, O + N * d, 0.0f);
    
    // Allocate buffers for l, m (scaling factors) 
    std::vector<float> l(N, 0.0f);    // ℓ = (0)_N ∈ R^N
    std::vector<float> m(N, -INFINITY); // m = (-∞)_N ∈ R^N
    
    // Device buffers
    sycl::buffer<float, 1> Q_buf(Q, sycl::range<1>(N * d));
    sycl::buffer<float, 1> K_buf(K, sycl::range<1>(N * d));
    sycl::buffer<float, 1> V_buf(V, sycl::range<1>(N * d));
    sycl::buffer<float, 1> O_buf(O, sycl::range<1>(N * d));
    sycl::buffer<float, 1> l_buf(l.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> m_buf(m.data(), sycl::range<1>(N));
    
    // Outer loop: for each block of K/V
    for (int j = 0; j < Tc; ++j) {
        q.submit([&](sycl::handler &h) {
            // Get accessors for all buffers
            auto K_acc = K_buf.get_access<sycl::access::mode::read>(h);
            auto V_acc = V_buf.get_access<sycl::access::mode::read>(h);
            auto Q_acc = Q_buf.get_access<sycl::access::mode::read>(h);
            auto O_acc = O_buf.get_access<sycl::access::mode::read_write>(h);
            auto l_acc = l_buf.get_access<sycl::access::mode::read_write>(h);
            auto m_acc = m_buf.get_access<sycl::access::mode::read_write>(h);
            
            h.parallel_for(sycl::range<1>(Tr), [=](sycl::id<1> i) {
                int i_start = i * Br;
                int j_start = j * Bc;
                
                // Compute S_ij = Q_i * K_j^T
                float S[Br][Bc] = {{0}};
                for (int qi = 0; qi < Br; ++qi) {
                    for (int kj = 0; kj < Bc; ++kj) {
                        for (int k = 0; k < d; ++k) {
                            int q_idx = (i_start + qi) * d + k;
                            int k_idx = (j_start + kj) * d + k;
                            if (q_idx < N * d && k_idx < N * d)
                                S[qi][kj] += Q_acc[q_idx] * K_acc[k_idx];
                        }
                    }
                }
                
                // Compute rowmax, exp, and rowsum
                float m_tilde[Br], l_tilde[Br];
                float P[Br][Bc] = {{0}};
                
                for (int qi = 0; qi < Br; ++qi) {
                    m_tilde[qi] = -INFINITY;
                    for (int kj = 0; kj < Bc; ++kj) {
                        m_tilde[qi] = std::max(m_tilde[qi], S[qi][kj]);
                    }
                }
                
                for (int qi = 0; qi < Br; ++qi) {
                    l_tilde[qi] = 0.0f;
                    for (int kj = 0; kj < Bc; ++kj) {
                        P[qi][kj] = std::exp(S[qi][kj] - m_tilde[qi]);
                        l_tilde[qi] += P[qi][kj];
                    }
                }
                
                // Update m, l, and O
                for (int qi = 0; qi < Br; ++qi) {
                    int q_idx = i_start + qi;
                    if (q_idx >= N) continue;
                    
                    float m_new = std::max(m_acc[q_idx], m_tilde[qi]);
                    float l_new = std::exp(m_acc[q_idx] - m_new) * l_acc[q_idx] +
                                  std::exp(m_tilde[qi] - m_new) * l_tilde[qi];
                    
                    // Write updated output O_i
                    for (int k = 0; k < d; ++k) {
                        float acc = 0.0f;
                        for (int kj = 0; kj < Bc; ++kj) {
                            int v_idx = (j_start + kj) * d + k;
                            if (v_idx < N * d)
                                acc += P[qi][kj] * V_acc[v_idx];
                        }
                        int o_idx = q_idx * d + k;
                        if (o_idx < N * d) {
                            float old_val = O_acc[o_idx];
                            float weighted = std::exp(m_acc[q_idx] - m_new) * old_val;
                            float new_val = (weighted + std::exp(m_tilde[qi] - m_new) * acc) / l_new;
                            O_acc[o_idx] = new_val;
                        }
                    }
                    
                    // Update l, m
                    l_acc[q_idx] = l_new;
                    m_acc[q_idx] = m_new;
                }
            });
        });
    }
}
