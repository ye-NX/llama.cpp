#pragma once

#include <CL/sycl.hpp>

template<typename T>
inline void run_flash_attn_block(
    sycl::accessor<T, 1, sycl::access::mode::read> k_acc,
    sycl::accessor<T, 1, sycl::access::mode::read> v_acc,
    sycl::accessor<T, 1, sycl::access::mode::read> q_acc,
    sycl::accessor<T, 1, sycl::access::mode::read_write> out_acc,
    sycl::accessor<T, 1, sycl::access::mode::read_write> l_acc,
    sycl::accessor<T, 1, sycl::access::mode::read_write> m_acc,
    const size_t i,
    const size_t j,
    const int Br,
    const int Bc,
    const int N,
    const int d,
    const bool masked,
    const float scale
) {
    // Block indices
    const int i_start = i * Br;
    const int j_start = j * Bc;
    
    // Shared memory for block multiplication
    T S[Br][Bc] = {{0}};
    
    // 1. Compute S = Q * K^T for this block
    for (int qi = 0; qi < Br; ++qi) {
        if (i_start + qi >= N) continue;
        
        for (int kj = 0; kj < Bc; ++kj) {
            if (j_start + kj >= N) continue;
            
            // Skip if masked attention
            if (masked && j_start + kj > i_start + qi) continue;
            
            T sum = 0;
            for (int k = 0; k < d; ++k) {
                int q_idx = (i_start + qi) * d + k;
                int k_idx = (j_start + kj) * d + k;
                sum += q_acc[q_idx] * k_acc[k_idx];
            }
            S[qi][kj] = sum * scale;
        }
    }
    
    // 2. Compute local maximum and exponentials
    T m_local[Br];
    T l_local[Br];
    T P[Br][Bc] = {{0}};
    
    for (int qi = 0; qi < Br; ++qi) {
        if (i_start + qi >= N) continue;
        
        // Find maximum in row
        m_local[qi] = -INFINITY;
        for (int kj = 0; kj < Bc; ++kj) {
            if (j_start + kj >= N) continue;
            if (masked && j_start + kj > i_start + qi) continue;
            m_local[qi] = sycl::max(m_local[qi], S[qi][kj]);
        }
        
        // Compute exponentials and local sum
        l_local[qi] = 0;
        for (int kj = 0; kj < Bc; ++kj) {
            if (j_start + kj >= N) continue;
            if (masked && j_start + kj > i_start + qi) continue;
            
            P[qi][kj] = sycl::exp(S[qi][kj] - m_local[qi]);
            l_local[qi] += P[qi][kj];
        }
    }
    
    // 3. Update accumulators and compute output
    for (int qi = 0; qi < Br; ++qi) {
        int q_idx = i_start + qi;
        if (q_idx >= N) continue;
        
        // Update m and l
        T m_old = m_acc[q_idx];
        T m_new = sycl::max(m_old, m_local[qi]);
        T l_new = sycl::exp(m_old - m_new) * l_acc[q_idx] + 
                  sycl::exp(m_local[qi] - m_new) * l_local[qi];
        
        // Update output
        for (int k = 0; k < d; ++k) {
            T acc = 0;
            for (int kj = 0; kj < Bc; ++kj) {
                if (j_start + kj >= N) continue;
                if (masked && j_start + kj > q_idx) continue;
                
                int v_idx = (j_start + kj) * d + k;
                acc += P[qi][kj] * v_acc[v_idx];
            }
            
            int o_idx = q_idx * d + k;
            T old_val = out_acc[o_idx];
            T weighted = sycl::exp(m_old - m_new) * old_val;
            out_acc[o_idx] = (weighted + sycl::exp(m_local[qi] - m_new) * acc) / l_new;
        }
        
        // Store updated l and m
        l_acc[q_idx] = l_new;
        m_acc[q_idx] = m_new;
    }
}