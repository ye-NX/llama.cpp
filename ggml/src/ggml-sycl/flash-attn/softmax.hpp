// softmax.hpp
#pragma once

#include <cmath>
#include <algorithm>

namespace flash_attention_softmax {

    // Compute row-wise maximum for numerical stability
    template<typename T>
    inline T compute_row_max(const T* values, int size) {
        T max_val = -INFINITY;
        for (int i = 0; i < size; ++i) {
            max_val = std::max(max_val, values[i]);
        }
        return max_val;
    }

    // Compute row-wise sum of exponentials (after subtracting max)
    template<typename T>
    inline T compute_exp_sum(const T* values, int size, T max_val) {
        T sum = 0.0;
        for (int i = 0; i < size; ++i) {
            sum += std::exp(values[i] - max_val);
        }
        return sum;
    }

    // Safe exponential function to prevent overflow
    template<typename T>
    inline T safe_exp(T x, T max_val) {
        return std::exp(x - max_val);
    }

    // Compute softmax in-place with numerical stability
    template<typename T>
    inline void softmax_inplace(T* values, int size) {
        T max_val = compute_row_max(values, size);
        T sum = compute_exp_sum(values, size, max_val);
        
        for (int i = 0; i < size; ++i) {
            values[i] = std::exp(values[i] - max_val) / sum;
        }
    }

    // Online softmax update for Flash Attention
    template<typename T>
    struct OnlineSoftmaxState {
        T max_val;
        T sum;
        
        OnlineSoftmaxState() : max_val(-INFINITY), sum(0.0) {}
        
        void update(T new_max, T new_sum) {
            T new_global_max = std::max(max_val, new_max);
            T new_global_sum = std::exp(max_val - new_global_max) * sum + 
                              std::exp(new_max - new_global_max) * new_sum;
            max_val = new_global_max;
            sum = new_global_sum;
        }
        
        T get_normalization_factor() const {
            return sum;
        }
    };

} // namespace flash_attention_softmax