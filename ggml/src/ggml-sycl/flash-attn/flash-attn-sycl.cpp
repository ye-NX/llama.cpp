#include "flash-attn-sycl.h"
#include "ggml-backend-sycl.h"
#include "kernels/flash-attn-kernel.h"

#include <assert.h>
#include <stdexcept>
#include <vector>

// Flash attention implementation
void ggml_sycl_flash_attn(
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    struct ggml_tensor * out,
    bool masked,
    float scale
) {
    GGML_ASSERT(q != nullptr);
    GGML_ASSERT(k != nullptr);
    GGML_ASSERT(v != nullptr);
    GGML_ASSERT(out != nullptr);
    
    // Get tensor dimensions
    const int64_t N = q->ne[1]; // sequence length
    const int64_t d = q->ne[0]; // head dimension
    
    // Block sizes (from Flash Attention paper)
    const int Bc = GGML_SYCL_FLASH_BLOCK_SIZE / (4 * d);  // Key/Value block size
    const int Br = std::min(Bc, (int)d);                  // Query block size
    
    // Number of blocks
    const int Tr = (N + Br - 1) / Br;  // Number of Q blocks
    const int Tc = (N + Bc - 1) / Bc;  // Number of K/V blocks
    
    try {
        sycl::queue& q = ggml_sycl_get_queue();
        
        // Create SYCL buffers for tensors
        sycl::buffer<float> q_buf(
            reinterpret_cast<float*>(q->data),
            sycl::range<1>(N * d)
        );
        sycl::buffer<float> k_buf(
            reinterpret_cast<float*>(k->data),
            sycl::range<1>(N * d)
        );
        sycl::buffer<float> v_buf(
            reinterpret_cast<float*>(v->data),
            sycl::range<1>(N * d)
        );
        sycl::buffer<float> out_buf(
            reinterpret_cast<float*>(out->data),
            sycl::range<1>(N * d)
        );
        
        // Allocate and initialize auxiliary buffers
        std::vector<float> l(N, 0.0f);
        std::vector<float> m(N, -INFINITY);
        
        sycl::buffer<float> l_buf(l.data(), sycl::range<1>(N));
        sycl::buffer<float> m_buf(m.data(), sycl::range<1>(N));
        
        // Main Flash Attention loop
        for (int j = 0; j < Tc; ++j) {
            q.submit([&](sycl::handler& h) {
                auto q_acc = q_buf.get_access<sycl::access::mode::read>(h);
                auto k_acc = k_buf.get_access<sycl::access::mode::read>(h);
                auto v_acc = v_buf.get_access<sycl::access::mode::read>(h);
                auto out_acc = out_buf.get_access<sycl::access::mode::read_write>(h);
                auto l_acc = l_buf.get_access<sycl::access::mode::read_write>(h);
                auto m_acc = m_buf.get_access<sycl::access::mode::read_write>(h);
                
                // Launch kernel
                h.parallel_for(sycl::range<1>(Tr), [=](sycl::id<1> i) {
                    run_flash_attn_block(
                        k_acc, v_acc, q_acc, out_acc,
                        l_acc, m_acc,
                        i[0], j,
                        Br, Bc, N, d,
                        masked,
                        scale
                    );
                });
            });
        }
        
    } catch (sycl::exception const& e) {
        fprintf(stderr, "SYCL exception in flash attention: %s\n", e.what());
        throw;
    }
}

// Initialize flash attention backend
void ggml_sycl_flash_attn_init(void) {
    // Register with ggml backend
    struct ggml_backend_sycl_flash_attn_context * ctx = 
        (struct ggml_backend_sycl_flash_attn_context *)malloc(sizeof(struct ggml_backend_sycl_flash_attn_context));
    
    ctx->compute_forward = ggml_sycl_flash_attn;
    
    ggml_backend_register(GGML_BACKEND_TYPE_SYCL_FLASH_ATTN, ctx);
}