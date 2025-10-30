#pragma once

#include "../common.hpp"

// Flash attention operation for SYCL backend
// This implements the Flash Attention algorithm optimized for SYCL devices
void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
