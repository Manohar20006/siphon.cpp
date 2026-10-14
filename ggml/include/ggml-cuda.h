#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Absolute prompt chunking for one backend (0 width disables): the next graphs process prompt
// positions [from, from + n_tokens) of a prompt ending at `end`; dense weight matmuls and flash
// attention reproduce the results of `width`-token ubatches starting at position 0.
GGML_BACKEND_API void ggml_backend_cuda_set_prompt_chunking(ggml_backend_t backend, int32_t width, int64_t from, int64_t n_tokens, int64_t end);

// Keep DirectStorage pooled MoE expert matmuls on MMQ at every batch size for one backend (default: batches of up to
// 8 tokens use MMVQ). Reproduces the numerics of builds before MMVQ handled expert pools, e.g. for decisions that are
// gated bit-exact against such references.
GGML_BACKEND_API void ggml_backend_cuda_set_pooled_moe_mmq(ggml_backend_t backend, bool enabled);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
