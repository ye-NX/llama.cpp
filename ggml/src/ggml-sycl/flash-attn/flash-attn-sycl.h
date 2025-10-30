#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Flash attention compute type
typedef struct ggml_tensor_flash_attn_sycl_params {
    bool masked;        // whether to apply casual mask
    float scale;        // scale factor for Q*K^T
    bool use_fp16;     // whether to use half precision
} ggml_tensor_flash_attn_sycl_params;

GGML_API void ggml_sycl_flash_attn(
    struct ggml_tensor * q,   // query
    struct ggml_tensor * k,   // key
    struct ggml_tensor * v,   // value
    struct ggml_tensor * out, // output
    bool masked,              // casual masking
    float scale              // scale factor for Q*K^T
);

// Backend initialization
GGML_API void ggml_sycl_flash_attn_init(void);

#ifdef __cplusplus
}
#endif