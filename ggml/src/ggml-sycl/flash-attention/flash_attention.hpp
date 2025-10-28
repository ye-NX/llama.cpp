// flash_attention.hpp
#pragma once

#include <CL/sycl.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

// Original function (deprecated - use flash_attention_sycl instead)
void flash_attention_forward(
    sycl::queue& q,
    float* Q, float* K, float* V, float* O,
    int N, int d
);

// Main Flash Attention implementation with memory efficiency
void flash_attention_sycl(
    sycl::queue &q,
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    const int N,
    const int d,
    const size_t M
);

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

            for (int k = 0; k < N; ++k) {
                float score = 0.0f;
                for (int d_i = 0; d_i < d; ++d_i) {
                    score += q_acc[i][d_i] * k_acc[k][d_i];
                }
                result += sycl::exp(score) * v_acc[k][j];
            }
            o_acc[i][j] = result; //  without normalization
        });
    });
}
