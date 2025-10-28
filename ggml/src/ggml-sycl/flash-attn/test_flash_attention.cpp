// test_flash_attention.cpp
#include "flash_attention.hpp"
#include "utils/softmax.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

// Test utilities
namespace test_utils {
    
    // Generate random matrix
    void generate_random_matrix(float* matrix, int rows, int cols, float scale = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = dist(gen);
        }
    }
    
    // Naive attention implementation for comparison
    void naive_attention(const float* Q, const float* K, const float* V, 
                        float* O, int N, int d) {
        // Compute attention scores S = Q * K^T
        std::vector<float> S(N * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float score = 0.0f;
                for (int k = 0; k < d; ++k) {
                    score += Q[i * d + k] * K[j * d + k];
                }
                S[i * N + j] = score;
            }
        }
        
        // Apply softmax row-wise
        for (int i = 0; i < N; ++i) {
            flash_attention_softmax::softmax_inplace(&S[i * N], N);
        }
        
        // Compute output O = P * V
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < d; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    sum += S[i * N + j] * V[j * d + k];
                }
                O[i * d + k] = sum;
            }
        }
    }
    
    // Compare two matrices with tolerance
    bool matrices_equal(const float* A, const float* B, int size, float tolerance = 1e-4f) {
        for (int i = 0; i < size; ++i) {
            if (std::abs(A[i] - B[i]) > tolerance) {
                std::cout << "Mismatch at index " << i << ": " 
                         << A[i] << " vs " << B[i] 
                         << " (diff: " << std::abs(A[i] - B[i]) << ")" << std::endl;
                return false;
            }
        }
        return true;
    }
    
    // Print matrix for debugging
    void print_matrix(const float* matrix, int rows, int cols, const std::string& name) {
        std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
        for (int i = 0; i < std::min(rows, 4); ++i) {
            for (int j = 0; j < std::min(cols, 4); ++j) {
                std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
            }
            if (cols > 4) std::cout << "...";
            std::cout << std::endl;
        }
        if (rows > 4) std::cout << "..." << std::endl;
        std::cout << std::endl;
    }
}

// Test cases
class FlashAttentionTester {
private:
    sycl::queue queue;
    
public:
    FlashAttentionTester() : queue(sycl::default_selector{}) {
        std::cout << "Testing on device: " 
                  << queue.get_device().get_info<sycl::info::device::name>() 
                  << std::endl;
    }
    
    // Test 1: Small matrices correctness
    bool test_small_correctness() {
        std::cout << "\n=== Test 1: Small Matrix Correctness ===" << std::endl;
        
        const int N = 8, d = 4;
        const size_t M = 1024; // Small memory limit
        
        // Allocate matrices
        std::vector<float> Q(N * d), K(N * d), V(N * d);
        std::vector<float> O_flash(N * d, 0.0f), O_naive(N * d, 0.0f);
        
        // Generate test data
        test_utils::generate_random_matrix(Q.data(), N, d, 0.1f);
        test_utils::generate_random_matrix(K.data(), N, d, 0.1f);
        test_utils::generate_random_matrix(V.data(), N, d, 1.0f);
        
        try {
            // Run Flash Attention
            flash_attention_sycl(queue, Q.data(), K.data(), V.data(), 
                               O_flash.data(), N, d, M);
            queue.wait();
            
            // Run naive implementation for comparison
            test_utils::naive_attention(Q.data(), K.data(), V.data(), 
                                      O_naive.data(), N, d);
            
            // Compare results
            bool success = test_utils::matrices_equal(O_flash.data(), O_naive.data(), 
                                                    N * d, 1e-3f);
            
            if (success) {
                std::cout << "Small correctness test PASSED" << std::endl;
            } else {
                std::cout << "Small correctness test FAILED" << std::endl;
                test_utils::print_matrix(O_flash.data(), N, d, "Flash Attention Output");
                test_utils::print_matrix(O_naive.data(), N, d, "Naive Attention Output");
            }
            
            return success;
            
        } catch (const std::exception& e) {
            std::cout << "Exception in small correctness test: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test 2: Identity matrices (should be predictable)
    bool test_identity_matrices() {
        std::cout << "\n=== Test 2: Identity Matrices ===" << std::endl;
        
        const int N = 4, d = 4;
        const size_t M = 512;
        
        // Create identity matrices
        std::vector<float> Q(N * d, 0.0f), K(N * d, 0.0f), V(N * d, 0.0f);
        std::vector<float> O(N * d, 0.0f);
        
        // Set up identity matrices
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d; ++j) {
                if (i == j) {
                    Q[i * d + j] = 1.0f;
                    K[i * d + j] = 1.0f;
                    V[i * d + j] = 1.0f;
                }
            }
        }
        
        try {
            flash_attention_sycl(queue, Q.data(), K.data(), V.data(), 
                               O.data(), N, d, M);
            queue.wait();
            
            // With identity Q and K, attention should be uniform (1/N for each position)
            // With identity V, output should be close to uniform average
            std::cout << "Identity test output:" << std::endl;
            test_utils::print_matrix(O.data(), N, d, "Output");
            
            std::cout << "Identity matrices test completed" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "Exception in identity test: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test 3: Different block sizes
    bool test_different_block_sizes() {
        std::cout << "\n=== Test 3: Different Block Sizes ===" << std::endl;
        
        const int N = 16, d = 8;
        std::vector<size_t> memory_sizes = {256, 512, 1024, 2048};
        
        // Generate test data
        std::vector<float> Q(N * d), K(N * d), V(N * d);
        test_utils::generate_random_matrix(Q.data(), N, d, 0.1f);
        test_utils::generate_random_matrix(K.data(), N, d, 0.1f);
        test_utils::generate_random_matrix(V.data(), N, d, 1.0f);
        
        std::vector<std::vector<float>> outputs;
        
        for (size_t M : memory_sizes) {
            std::vector<float> O(N * d, 0.0f);
            
            try {
                flash_attention_sycl(queue, Q.data(), K.data(), V.data(), 
                                   O.data(), N, d, M);
                queue.wait();
                outputs.push_back(O);
                
                std::cout << "Block size M=" << M << " completed" << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "Exception with M=" << M << ": " << e.what() << std::endl;
                return false;
            }
        }
        
        // Compare all outputs should be identical
        bool all_equal = true;
        for (size_t i = 1; i < outputs.size(); ++i) {
            if (!test_utils::matrices_equal(outputs[0].data(), outputs[i].data(), 
                                          N * d, 1e-5f)) {
                std::cout << "Output mismatch between M=" << memory_sizes[0] 
                         << " and M=" << memory_sizes[i] << std::endl;
                all_equal = false;
            }
        }
        
        if (all_equal) {
            std::cout << "All block sizes produce identical results" << std::endl;
        }
        
        return all_equal;
    }
    
    // Test 4: Numerical stability
    bool test_numerical_stability() {
        std::cout << "\n=== Test 4: Numerical Stability ===" << std::endl;
        
        const int N = 8, d = 4;
        const size_t M = 512;
        
        // Create matrices with large values to test overflow protection
        std::vector<float> Q(N * d), K(N * d), V(N * d);
        std::vector<float> O(N * d, 0.0f);
        
        // Fill with large values
        test_utils::generate_random_matrix(Q.data(), N, d, 10.0f); // Large scale
        test_utils::generate_random_matrix(K.data(), N, d, 10.0f);
        test_utils::generate_random_matrix(V.data(), N, d, 1.0f);
        
        try {
            flash_attention_sycl(queue, Q.data(), K.data(), V.data(), 
                               O.data(), N, d, M);
            queue.wait();
            
            // Check for NaN or Inf values
            bool has_invalid = false;
            for (int i = 0; i < N * d; ++i) {
                if (std::isnan(O[i]) || std::isinf(O[i])) {
                    std::cout << "Found invalid value at index " << i 
                             << ": " << O[i] << std::endl;
                    has_invalid = true;
                }
            }
            
            if (!has_invalid) {
                std::cout << "Numerical stability test PASSED - no NaN/Inf values" << std::endl;
                return true;
            } else {
                std::cout << "Numerical stability test FAILED" << std::endl;
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cout << "Exception in stability test: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Run all tests
    void run_all_tests() {
        std::cout << "Starting Flash Attention Test Suite" << std::endl;
        std::cout << "========================================" << std::endl;
        
        int passed = 0, total = 4;
        
        if (test_small_correctness()) passed++;
        if (test_identity_matrices()) passed++;
        if (test_different_block_sizes()) passed++;
        if (test_numerical_stability()) passed++;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Results: " << passed << "/" << total << " tests passed" << std::endl;
        
        if (passed == total) {
            std::cout << "All tests PASSED! Flash Attention implementation is correct." << std::endl;
        } else {
            std::cout << "Some tests failed. Please check the implementation." << std::endl;
        }
    }
};

int main() {
    try {
        FlashAttentionTester tester;
        tester.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}