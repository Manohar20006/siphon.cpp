// Expert storage loader for Ollama.
// Windows: Microsoft DirectStorage. Linux: NVIDIA GPUDirect Storage/cuFile.

#ifndef DSTORAGE_LOADER_H
#define DSTORAGE_LOADER_H

#include <stdint.h>
#include <stddef.h>

#define DS_API  // statically linked, no import/export needed

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DSLoader* DSLoaderHandle;

// --- Availability & lifecycle ---
DS_API int ds_loader_available();
DS_API int32_t ds_loader_get_hresult(); // returns last HRESULT for debugging
DS_API DSLoaderHandle ds_loader_create();
DS_API void ds_loader_destroy(DSLoaderHandle loader);

// --- GPU buffer management ---
// Creates a D3D12 committed resource (DEFAULT heap) suitable for DirectStorage writes.
// Returns opaque pointer to ID3D12Resource, or NULL on failure.
DS_API void* ds_loader_create_gpu_buffer(DSLoaderHandle loader, uint64_t size);

// Destroys a GPU buffer created by ds_loader_create_gpu_buffer.
DS_API void ds_loader_destroy_gpu_buffer(void* gpu_buffer);

// --- File reads ---
// Read file data directly to a GPU buffer (SSD -> GPU, bypasses CPU).
// gpu_buffer must be created by ds_loader_create_gpu_buffer.
DS_API int ds_loader_read(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer
);

// Read file data directly to a GPU buffer with automatic chunking.
// Splits reads > 32MB into multiple DirectStorage requests, enqueues all,
// then submits once and waits once. No size limit.
DS_API int ds_loader_read_chunked(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t total_size,
    void* gpu_buffer
);

// Read file data to CPU memory via DirectStorage.
// Useful for testing and for data that needs CPU processing.
DS_API int ds_loader_read_to_memory(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* dest_memory
);

// --- Batched reads (file handle caching + multi-request submit) ---

// Open a file and cache the IDStorageFile handle inside the loader.
// Subsequent ds_loader_enqueue_read calls use this cached handle.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_open_file(
    DSLoaderHandle loader,
    const wchar_t* file_path
);

// Close/release the cached file handle.
DS_API void ds_loader_close_file(DSLoaderHandle loader);

// Enqueue a single read request using the cached file handle.
// Does NOT submit — call ds_loader_submit_and_wait after enqueuing all requests.
// Automatically splits reads > 32MB into multiple chunked requests.
// buffer_offset is the offset within the gpu_buffer to write to.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_enqueue_read(
    DSLoaderHandle loader,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer,
    uint64_t buffer_offset
);

// Submit all enqueued requests and wait for completion.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_submit_and_wait(DSLoaderHandle loader);

// --- Async submit for prefetching ---

// Submit all enqueued requests WITHOUT waiting. Returns immediately.
// The DMA transfer runs in the background. Call ds_loader_wait_complete()
// or ds_loader_is_complete() to check/wait for completion.
// If a previous async submit is still pending, waits for it first.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_submit(DSLoaderHandle loader);

// Non-blocking check: returns 1 if the last ds_loader_submit() has completed
// (or if there is no pending work), 0 if still in-flight.
DS_API int ds_loader_is_complete(DSLoaderHandle loader);

// Block until the last ds_loader_submit() completes.
// Returns 0 on success, -1 on failure (DirectStorage error in the batch).
DS_API int ds_loader_wait_complete(DSLoaderHandle loader);

// --- GPU readback (for verification/testing) ---
// Copy data from GPU buffer back to CPU memory.
// Uses a D3D12 readback heap + command list internally.
DS_API int ds_loader_gpu_readback(
    DSLoaderHandle loader,
    void* gpu_buffer,
    uint64_t size,
    void* dest_memory
);

// --- Diagnostic for shared heap support ---
DS_API int ds_loader_debug_shared(DSLoaderHandle loader);

// --- CUDA interop (D3D12 <-> CUDA shared memory) ---
// Bridges D3D12 GPU buffers to CUDA device pointers via nvcuda.dll.
// All CUDA functions are loaded dynamically — no CUDA SDK required.
// nvcuda.dll ships with every NVIDIA display driver installation.

typedef struct CUDAInterop* CUDAInteropHandle;

// Check if CUDA is available (nvcuda.dll loadable, cuInit succeeds).
// Returns 1 if available, 0 if not.
DS_API int ds_loader_cuda_available();

// Creates a D3D12 buffer with D3D12_HEAP_FLAG_SHARED, suitable for
// both DirectStorage writes and CUDA interop export.
// Returns opaque pointer to ID3D12Resource, or NULL on failure.
DS_API void* ds_loader_create_shared_gpu_buffer(DSLoaderHandle loader, uint64_t size);

// Exports a shared D3D12 GPU buffer to CUDA.
// Creates a shared NT handle, imports into CUDA via cuImportExternalMemory,
// maps to a CUDA device pointer via cuExternalMemoryGetMappedBuffer.
// The gpu_buffer MUST have been created by ds_loader_create_shared_gpu_buffer.
// Returns an opaque interop handle, or NULL on failure.
DS_API CUDAInteropHandle ds_loader_export_to_cuda(
    DSLoaderHandle loader,
    void* shared_gpu_buffer,
    uint64_t size
);

// Returns the CUDA device pointer (CUdeviceptr as uint64) from an interop handle.
// This pointer can be passed to CUDA/GGML compute kernels.
DS_API uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle interop);

// Copies data from the CUDA device pointer to host memory (for verification).
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle interop, void* dest, uint64_t size);

// Destroys a CUDA interop handle, releasing the CUDA device pointer,
// external memory object, and shared NT handle.
DS_API void ds_loader_cuda_destroy(CUDAInteropHandle interop);

// --- Stream-to-CUDA (the integration point for Ollama) ---

// Loads file data directly to a CUDA device pointer in one call.
// Uses a reusable internal staging buffer (D3D12 shared heap + CUDA interop).
// Path: SSD -> DirectStorage DMA -> staging GPU buffer -> cuMemcpyDtoD -> dest.
// The staging buffer auto-grows as needed and is reused across calls.
// cuda_dest_ptr is a CUdeviceptr (e.g., from ggml_tensor->data or cuMemAlloc).
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_stream_to_cuda(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    uint64_t cuda_dest_ptr
);

typedef struct DSLoaderStreamRequest {
    const wchar_t* file_path;
    uint64_t file_offset;
    uint64_t size;
    uint64_t cuda_dest_ptr;
    uint64_t uncompressed_size; // uncompressed output size, larger than size if GDeflate compressed
} DSLoaderStreamRequest;

typedef struct DSLoaderHostToCudaRequest {
    const void* host_src_ptr;
    uint64_t size;
    uint64_t cuda_dest_ptr;
} DSLoaderHostToCudaRequest;

typedef struct DSLoaderCudaToHostRequest {
    uint64_t cuda_src_ptr;
    void* host_dest_ptr;
    uint64_t size;
} DSLoaderCudaToHostRequest;

typedef struct DSLoaderIoStats {
    uint64_t batches;
    uint64_t requests;
    uint64_t bytes;
    uint64_t wall_us;
    uint64_t read_sum_us;
    uint64_t read_max_us;
    uint64_t batch_api_attempts;
    uint64_t batch_api_successes;
    uint64_t worker_batches;
    uint64_t serial_batches;
    uint64_t aligned_4k_requests;
    uint64_t unaligned_4k_requests;
    uint64_t file_offset_4k_aligned;
    uint64_t size_4k_aligned;
    uint64_t cuda_dest_4k_aligned;
    uint64_t file_offset_mod_min;
    uint64_t file_offset_mod_max;
    uint64_t file_offset_mod_gcd;
    uint64_t size_mod_min;
    uint64_t size_mod_max;
    uint64_t size_mod_gcd;
    uint64_t cuda_dest_mod_min;
    uint64_t cuda_dest_mod_max;
    uint64_t cuda_dest_mod_gcd;
    uint64_t size_le_64k;
    uint64_t size_le_256k;
    uint64_t size_le_1m;
    uint64_t size_le_4m;
    uint64_t size_gt_4m;
    uint64_t min_request_bytes;
    uint64_t max_request_bytes;
    uint64_t max_batch_requests;
    uint64_t max_batch_bytes;
    uint64_t max_batch_wall_us;
} DSLoaderIoStats;

// Loads multiple file slices directly to CUDA device pointers with one
// DirectStorage queue submit/wait. Each slice is staged into a separate region
// of the reusable staging buffer, then copied to its final CUDA destination.
DS_API int ds_loader_stream_to_cuda_batch(
    DSLoaderHandle loader,
    const DSLoaderStreamRequest* requests,
    int request_count
);

// Returns cumulative loader-side I/O shape/timing counters since loader
// creation. Returns 0 on success, -1 on invalid arguments.
DS_API int ds_loader_get_io_stats(DSLoaderHandle loader, DSLoaderIoStats* out_stats);

// Allocates host memory suitable for host-to-device transfer caching.
// Uses CUDA pinned host memory when available, otherwise falls back to aligned RAM.
// out_is_pinned receives 1 for CUDA-pinned memory, 0 for fallback RAM.
DS_API void* ds_loader_host_alloc(uint64_t size, int* out_is_pinned);

// Frees memory returned by ds_loader_host_alloc.
DS_API void ds_loader_host_free(void* ptr, int is_pinned);

// Copies a batch of host slices to CUDA device pointers.
DS_API int ds_loader_host_to_cuda_batch(
    DSLoaderHandle loader,
    const DSLoaderHostToCudaRequest* requests,
    int request_count
);

// Copies a batch of CUDA device slices to pinned host memory.
DS_API int ds_loader_cuda_to_host_batch(
    DSLoaderHandle loader,
    const DSLoaderCudaToHostRequest* requests,
    int request_count
);

// Allocate a CUDA device buffer via cuMemAlloc (for testing).
// Returns CUdeviceptr as uint64, or 0 on failure.
DS_API uint64_t ds_loader_cuda_alloc(uint64_t size);

// Free a CUDA device buffer allocated by ds_loader_cuda_alloc.
DS_API void ds_loader_cuda_free(uint64_t ptr);

// Wait for any pending async CUDA operations on the loader's copy stream to complete.
DS_API int ds_loader_cuda_wait_event(DSLoaderHandle loader);

// Copy from a raw CUDA device pointer to host memory (for testing).
// src_cuda_ptr is a CUdeviceptr (from cuMemAlloc or ggml tensor->data).
DS_API int ds_loader_cuda_dtoh(uint64_t src_cuda_ptr, void* dest, uint64_t size);

// Copy between raw CUDA device pointers.
DS_API int ds_loader_cuda_dtod(uint64_t dst_cuda_ptr, uint64_t src_cuda_ptr, uint64_t size);

// --- GDeflate Compression Utility API ---
DS_API int ds_loader_compress_buffer(
    const void* src,
    uint64_t src_size,
    void* dst,
    uint64_t dst_size,
    uint64_t* out_compressed_size
);

#ifdef __cplusplus
}
#endif

#endif // DSTORAGE_LOADER_H
