// flash_kernel.hpp
#pragma once

#include <CL/sycl.hpp>
#include <cmath>
#include <algorithm>

// Legacy kernel function - deprecated
// This is a naive implementation that doesn't use Flash Attention algorithm
void run_flash_attention_kernel(
    sycl::queue& q,
    sycl::buffer<float, 2>& Q,
    sycl::buffer<float, 2>& K,
    sycl::buffer<float, 2>& V,
    sycl::buffer<float, 2>& O,
    int N, int d
) {
    q.submit([&](sycl::handler& h) {
        auto q_acc = Q.get_access<sycl::access::mode::read>(h);
        auto k_acc = K.get_access<sycl::access::mode::read>(h);
        auto v_acc = V.get_access<sycl::access::mode::read>(h);
        auto o_acc = O.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>(N, d), [=](sycl::id<2> idx) {
            int i = idx[0], j = idx[1];
            float result = 0.0f;

            // WARNING: This is NOT Flash Attention - it's a naive implementation
            // without proper softmax normalization and memory efficiency
            for (int k = 0; k < N; ++k) {
                float score = 0.0f;
                for (int d_i = 0; d_i < d; ++d_i) {
                    score += q_acc[i][d_i] * k_acc[k][d_i];
                }
                result += sycl::exp(score) * v_acc[k][j];
            }
            o_acc[i][j] = result; // without normalization - INCORRECT!
        });
    });
}

// Utility functions for Flash Attention algorithm
namespace flash_attention_utils {
    
    // Safe exponential function to prevent overflow
    inline float safe_exp(float x, float max_val) {
        return std::exp(x - max_val);
    }
    
    // Compute maximum value in a row of scores
    template<int BLOCK_SIZE>
    inline float compute_row_max(float scores[BLOCK_SIZE]) {
        float max_val = -INFINITY;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            max_val = std::max(max_val, scores[i]);
        }
        return max_val;
    }
    
    // Compute sum of probabilities after exponential
    template<int BLOCK_SIZE>
    inline float compute_row_sum(float probs[BLOCK_SIZE]) {
        float sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += probs[i];
        }
        return sum;
    }
}
