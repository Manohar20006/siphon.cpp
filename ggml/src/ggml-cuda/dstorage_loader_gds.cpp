// Linux GPUDirect Storage implementation of the ds_loader_* ABI.
// Uses dynamic loading so llama can still build and report cleanly when GDS
// is not installed on the host.

#if defined(__linux__)

#include "dstorage_loader.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <atomic>
#include <numeric>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

typedef int CUresult;
typedef int CUdevice;
typedef void * CUcontext;
typedef void * CUstream;
typedef unsigned long long CUdeviceptr;

static constexpr CUresult CUDA_SUCCESS = 0;
static constexpr int CU_FILE_SUCCESS = 0;
static constexpr int CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1;
static constexpr int CUFILE_READ = 0;
static constexpr int CUFILE_BATCH = 1;
static constexpr int CUFILE_COMPLETE = 0x0000010;

struct CUfileError_t {
    int err;
    CUresult cu_err;
};

struct CUfileDescr_t {
    int type;
    int reserved;
    union {
        int fd;
        void * handle;
    } handle;
    void * fs_ops;
};

typedef void * CUfileHandle_t;
typedef void * CUfileBatchHandle_t;

struct CUfileIOParams_t {
    int mode;
    union {
        struct {
            void * devPtr_base;
            off_t file_offset;
            off_t devPtr_offset;
            size_t size;
        } batch;
    } u;
    CUfileHandle_t fh;
    int opcode;
    void * cookie;
};

struct CUfileIOEvents_t {
    void * cookie;
    int status;
    size_t ret;
};

typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*PFN_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*PFN_cuCtxSynchronize)(void);
typedef CUresult (*PFN_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemcpyDtoD_v2)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
typedef CUresult (*PFN_cuMemcpyDtoHAsync_v2)(void *, CUdeviceptr, size_t, CUstream);
typedef CUresult (*PFN_cuMemcpyHtoDAsync_v2)(CUdeviceptr, const void *, size_t, CUstream);
typedef CUresult (*PFN_cuMemAlloc_v2)(CUdeviceptr *, size_t);
typedef CUresult (*PFN_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*PFN_cuMemHostAlloc_v2)(void **, size_t, unsigned int);
typedef CUresult (*PFN_cuMemFreeHost)(void *);
typedef CUresult (*PFN_cuStreamCreate)(CUstream *, unsigned int);
typedef CUresult (*PFN_cuStreamDestroy_v2)(CUstream);
typedef CUresult (*PFN_cuStreamSynchronize)(CUstream);

typedef CUfileError_t (*PFN_cuFileDriverOpen)(void);
typedef CUfileError_t (*PFN_cuFileDriverClose)(void);
typedef CUfileError_t (*PFN_cuFileHandleRegister)(CUfileHandle_t *, CUfileDescr_t *);
typedef void (*PFN_cuFileHandleDeregister)(CUfileHandle_t);
typedef CUfileError_t (*PFN_cuFileBufRegister)(const void *, size_t, int);
typedef void (*PFN_cuFileBufDeregister)(const void *);
typedef ssize_t (*PFN_cuFileRead)(CUfileHandle_t, void *, size_t, off_t, off_t);
typedef CUfileError_t (*PFN_cuFileBatchIOSetUp)(CUfileBatchHandle_t *, unsigned);
typedef CUfileError_t (*PFN_cuFileBatchIOSubmit)(
        CUfileBatchHandle_t, unsigned, CUfileIOParams_t *, unsigned int);
typedef CUfileError_t (*PFN_cuFileBatchIOGetStatus)(
        CUfileBatchHandle_t, unsigned, unsigned *, CUfileIOEvents_t *, struct timespec *);
typedef CUfileError_t (*PFN_cuFileBatchIOCancel)(CUfileBatchHandle_t);
typedef void (*PFN_cuFileBatchIODestroy)(CUfileBatchHandle_t);

static void * g_cuda_module = nullptr;
static void * g_cufile_module = nullptr;

static PFN_cuInit                   g_cuInit = nullptr;
static PFN_cuDeviceGet              g_cuDeviceGet = nullptr;
static PFN_cuDevicePrimaryCtxRetain g_cuDevicePrimaryCtxRetain = nullptr;
static PFN_cuCtxSetCurrent          g_cuCtxSetCurrent = nullptr;
static PFN_cuCtxSynchronize         g_cuCtxSynchronize = nullptr;
static PFN_cuMemcpyDtoH_v2          g_cuMemcpyDtoH = nullptr;
static PFN_cuMemcpyDtoD_v2          g_cuMemcpyDtoD = nullptr;
static PFN_cuMemcpyHtoD_v2          g_cuMemcpyHtoD = nullptr;
static PFN_cuMemcpyDtoHAsync_v2     g_cuMemcpyDtoHAsync = nullptr;
static PFN_cuMemcpyHtoDAsync_v2     g_cuMemcpyHtoDAsync = nullptr;
static PFN_cuMemAlloc_v2            g_cuMemAlloc = nullptr;
static PFN_cuMemFree_v2             g_cuMemFree = nullptr;
static PFN_cuMemHostAlloc_v2        g_cuMemHostAlloc = nullptr;
static PFN_cuMemFreeHost            g_cuMemFreeHost = nullptr;
static PFN_cuStreamCreate           g_cuStreamCreate = nullptr;
static PFN_cuStreamDestroy_v2       g_cuStreamDestroy = nullptr;
static PFN_cuStreamSynchronize      g_cuStreamSynchronize = nullptr;

static PFN_cuFileDriverOpen         g_cuFileDriverOpen = nullptr;
static PFN_cuFileDriverClose        g_cuFileDriverClose = nullptr;
static PFN_cuFileHandleRegister     g_cuFileHandleRegister = nullptr;
static PFN_cuFileHandleDeregister   g_cuFileHandleDeregister = nullptr;
static PFN_cuFileBufRegister        g_cuFileBufRegister = nullptr;
static PFN_cuFileBufDeregister      g_cuFileBufDeregister = nullptr;
static PFN_cuFileRead               g_cuFileRead = nullptr;
static PFN_cuFileBatchIOSetUp       g_cuFileBatchIOSetUp = nullptr;
static PFN_cuFileBatchIOSubmit      g_cuFileBatchIOSubmit = nullptr;
static PFN_cuFileBatchIOGetStatus   g_cuFileBatchIOGetStatus = nullptr;
static PFN_cuFileBatchIOCancel      g_cuFileBatchIOCancel = nullptr;
static PFN_cuFileBatchIODestroy     g_cuFileBatchIODestroy = nullptr;

static bool g_cuda_initialized = false;
static bool g_cuda_attempted = false;
static bool g_cufile_initialized = false;
static bool g_cufile_attempted = false;
static CUcontext g_cuda_context = nullptr;
static CUdevice g_cuda_device = 0;
static int32_t g_last_error = 0;
static std::mutex g_init_mutex;

struct RegisteredCudaBuffer {
    uint64_t ptr = 0;
    uint64_t size = 0;
};

static std::mutex g_registered_buffers_mutex;
static std::vector<RegisteredCudaBuffer> g_registered_buffers;

static void set_last_error(int32_t err) {
    g_last_error = err;
}

static bool gds_register_per_request() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_GDS_REGISTER_PER_REQUEST");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static int gds_read_threads() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_GDS_READ_THREADS");
        if (env == nullptr || env[0] == '\0') {
            return 20;
        }
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env || *end != '\0') {
            return 20;
        }
        return int(std::max<long>(1, std::min<long>(parsed, 32)));
    }();
    return value;
}

static bool gds_use_batch() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_GDS_USE_BATCH");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static int gds_worker_claim_count() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_GDS_WORK_CLAIM");
        if (env == nullptr || env[0] == '\0') {
            return 1;
        }
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env || *end != '\0') {
            return 1;
        }
        return int(std::max<long>(1, std::min<long>(parsed, 16)));
    }();
    return value;
}

static bool gds_aligned_staging_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_GDS_ALIGNED_STAGING");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool gds_stage_unaligned_dest_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_GDS_STAGE_UNALIGNED_DEST");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool gds_thread_file_handles_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_GDS_THREAD_FILE_HANDLES");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool gds_plain_read_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_GDS_PLAIN_READ");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

template <typename T>
static bool load_symbol(void * module, const char * name, T & out) {
    out = reinterpret_cast<T>(dlsym(module, name));
    return out != nullptr;
}

static bool ensure_cuda_loaded() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_cuda_initialized) {
        const CUresult cr = g_cuCtxSetCurrent(g_cuda_context);
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return false;
        }
        return true;
    }
    if (g_cuda_attempted) {
        return false;
    }
    g_cuda_attempted = true;

    g_cuda_module = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (g_cuda_module == nullptr) {
        g_cuda_module = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (g_cuda_module == nullptr) {
        set_last_error(-ENOENT);
        return false;
    }

    if (!load_symbol(g_cuda_module, "cuInit", g_cuInit) ||
        !load_symbol(g_cuda_module, "cuDeviceGet", g_cuDeviceGet) ||
        !load_symbol(g_cuda_module, "cuDevicePrimaryCtxRetain", g_cuDevicePrimaryCtxRetain) ||
        !load_symbol(g_cuda_module, "cuCtxSetCurrent", g_cuCtxSetCurrent) ||
        !load_symbol(g_cuda_module, "cuCtxSynchronize", g_cuCtxSynchronize) ||
        !load_symbol(g_cuda_module, "cuMemcpyDtoH_v2", g_cuMemcpyDtoH) ||
        !load_symbol(g_cuda_module, "cuMemcpyDtoD_v2", g_cuMemcpyDtoD) ||
        !load_symbol(g_cuda_module, "cuMemcpyHtoD_v2", g_cuMemcpyHtoD) ||
        !load_symbol(g_cuda_module, "cuMemAlloc_v2", g_cuMemAlloc) ||
        !load_symbol(g_cuda_module, "cuMemFree_v2", g_cuMemFree) ||
        !load_symbol(g_cuda_module, "cuMemHostAlloc", g_cuMemHostAlloc) ||
        !load_symbol(g_cuda_module, "cuMemFreeHost", g_cuMemFreeHost)) {
        set_last_error(-ENOSYS);
        return false;
    }

    load_symbol(g_cuda_module, "cuMemcpyDtoHAsync_v2", g_cuMemcpyDtoHAsync);
    load_symbol(g_cuda_module, "cuMemcpyHtoDAsync_v2", g_cuMemcpyHtoDAsync);
    load_symbol(g_cuda_module, "cuStreamCreate", g_cuStreamCreate);
    load_symbol(g_cuda_module, "cuStreamDestroy_v2", g_cuStreamDestroy);
    load_symbol(g_cuda_module, "cuStreamSynchronize", g_cuStreamSynchronize);

    CUresult cr = g_cuInit(0);
    if (cr != CUDA_SUCCESS) {
        set_last_error(cr);
        return false;
    }
    cr = g_cuDeviceGet(&g_cuda_device, 0);
    if (cr != CUDA_SUCCESS) {
        set_last_error(cr);
        return false;
    }
    cr = g_cuDevicePrimaryCtxRetain(&g_cuda_context, g_cuda_device);
    if (cr != CUDA_SUCCESS) {
        set_last_error(cr);
        return false;
    }
    cr = g_cuCtxSetCurrent(g_cuda_context);
    if (cr != CUDA_SUCCESS) {
        set_last_error(cr);
        return false;
    }

    g_cuda_initialized = true;
    return true;
}

static bool ensure_cufile_loaded() {
    if (!ensure_cuda_loaded()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_cufile_initialized) {
        return true;
    }
    if (g_cufile_attempted) {
        return false;
    }
    g_cufile_attempted = true;

    g_cufile_module = dlopen("libcufile.so", RTLD_NOW | RTLD_LOCAL);
    if (g_cufile_module == nullptr) {
        g_cufile_module = dlopen("libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
    }
    if (g_cufile_module == nullptr) {
        set_last_error(-ENOENT);
        return false;
    }

    if (!load_symbol(g_cufile_module, "cuFileDriverOpen", g_cuFileDriverOpen) ||
        !load_symbol(g_cufile_module, "cuFileDriverClose", g_cuFileDriverClose) ||
        !load_symbol(g_cufile_module, "cuFileHandleRegister", g_cuFileHandleRegister) ||
        !load_symbol(g_cufile_module, "cuFileHandleDeregister", g_cuFileHandleDeregister) ||
        !load_symbol(g_cufile_module, "cuFileBufRegister", g_cuFileBufRegister) ||
        !load_symbol(g_cufile_module, "cuFileBufDeregister", g_cuFileBufDeregister) ||
        !load_symbol(g_cufile_module, "cuFileRead", g_cuFileRead)) {
        set_last_error(-ENOSYS);
        return false;
    }

    load_symbol(g_cufile_module, "cuFileBatchIOSetUp", g_cuFileBatchIOSetUp);
    load_symbol(g_cufile_module, "cuFileBatchIOSubmit", g_cuFileBatchIOSubmit);
    load_symbol(g_cufile_module, "cuFileBatchIOGetStatus", g_cuFileBatchIOGetStatus);
    load_symbol(g_cufile_module, "cuFileBatchIOCancel", g_cuFileBatchIOCancel);
    load_symbol(g_cufile_module, "cuFileBatchIODestroy", g_cuFileBatchIODestroy);

    const CUfileError_t status = g_cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        set_last_error(status.err != 0 ? status.err : status.cu_err);
        return false;
    }

    g_cufile_initialized = true;
    return true;
}

static bool find_registered_cuda_buffer(
        uint64_t ptr,
        uint64_t size,
        RegisteredCudaBuffer & out_buffer) {
    std::lock_guard<std::mutex> lock(g_registered_buffers_mutex);
    for (const RegisteredCudaBuffer & buffer : g_registered_buffers) {
        const uint64_t begin = buffer.ptr;
        const uint64_t end = buffer.ptr + buffer.size;
        if (ptr >= begin && size <= end - ptr) {
            out_buffer = buffer;
            return true;
        }
    }
    return false;
}

static void remember_registered_cuda_buffer(uint64_t ptr, uint64_t size) {
    std::lock_guard<std::mutex> lock(g_registered_buffers_mutex);
    g_registered_buffers.push_back({ ptr, size });
}

static bool forget_registered_cuda_buffer(uint64_t ptr) {
    std::lock_guard<std::mutex> lock(g_registered_buffers_mutex);
    for (auto it = g_registered_buffers.begin(); it != g_registered_buffers.end(); ++it) {
        if (it->ptr == ptr) {
            g_registered_buffers.erase(it);
            return true;
        }
    }
    return false;
}

static std::string narrow_path(const wchar_t * path) {
    std::string out;
    if (path == nullptr) {
        return out;
    }
    while (*path != 0) {
        const wchar_t ch = *path++;
        out.push_back(ch >= 0 && ch <= 0x7f ? char(ch) : '?');
    }
    return out;
}

struct FileHandle {
    int fd = -1;
    CUfileHandle_t cf_handle = nullptr;
};

struct DSLoader {
    std::mutex mutex;
    std::unordered_map<std::string, FileHandle> files;

    std::mutex work_mutex;
    std::condition_variable work_cv;
    std::condition_variable done_cv;
    std::vector<std::thread> read_workers;
    const DSLoaderStreamRequest * work_requests = nullptr;
    int work_request_count = 0;
    std::atomic<int> work_next_index{0};
    std::atomic<int> work_remaining{0};
    std::atomic<int> work_failed{0};
    std::atomic<uint64_t> work_read_us{0};
    std::atomic<uint64_t> work_read_bytes{0};
    std::atomic<uint64_t> work_read_calls{0};
    std::atomic<uint64_t> work_read_max_us{0};
    bool work_stop = false;
    uint64_t work_generation = 0;
    uint64_t work_completed_generation = 0;

    std::atomic<uint64_t> io_batches{0};
    std::atomic<uint64_t> io_requests{0};
    std::atomic<uint64_t> io_bytes{0};
    std::atomic<uint64_t> io_wall_us{0};
    std::atomic<uint64_t> io_read_sum_us{0};
    std::atomic<uint64_t> io_read_max_us{0};
    std::atomic<uint64_t> io_batch_api_attempts{0};
    std::atomic<uint64_t> io_batch_api_successes{0};
    std::atomic<uint64_t> io_worker_batches{0};
    std::atomic<uint64_t> io_serial_batches{0};
    std::atomic<uint64_t> io_aligned_4k_requests{0};
    std::atomic<uint64_t> io_unaligned_4k_requests{0};
    std::atomic<uint64_t> io_file_offset_4k_aligned{0};
    std::atomic<uint64_t> io_size_4k_aligned{0};
    std::atomic<uint64_t> io_cuda_dest_4k_aligned{0};
    std::atomic<uint64_t> io_file_offset_mod_min{UINT64_MAX};
    std::atomic<uint64_t> io_file_offset_mod_max{0};
    std::atomic<uint64_t> io_file_offset_mod_gcd{0};
    std::atomic<uint64_t> io_size_mod_min{UINT64_MAX};
    std::atomic<uint64_t> io_size_mod_max{0};
    std::atomic<uint64_t> io_size_mod_gcd{0};
    std::atomic<uint64_t> io_cuda_dest_mod_min{UINT64_MAX};
    std::atomic<uint64_t> io_cuda_dest_mod_max{0};
    std::atomic<uint64_t> io_cuda_dest_mod_gcd{0};
    std::atomic<uint64_t> io_size_le_64k{0};
    std::atomic<uint64_t> io_size_le_256k{0};
    std::atomic<uint64_t> io_size_le_1m{0};
    std::atomic<uint64_t> io_size_le_4m{0};
    std::atomic<uint64_t> io_size_gt_4m{0};
    std::atomic<uint64_t> io_min_request_bytes{UINT64_MAX};
    std::atomic<uint64_t> io_max_request_bytes{0};
    std::atomic<uint64_t> io_max_batch_requests{0};
    std::atomic<uint64_t> io_max_batch_bytes{0};
    std::atomic<uint64_t> io_max_batch_wall_us{0};

    std::mutex h2d_mutex;
    std::mutex d2h_mutex;
    CUstream h2d_stream = nullptr;
    CUstream d2h_stream = nullptr;
};

static uint64_t gds_now_us() {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool gds_io_profile_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_GDS_IO_PROFILE");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static void gds_atomic_max(std::atomic<uint64_t> & value, uint64_t candidate) {
    uint64_t current = value.load(std::memory_order_relaxed);
    while (current < candidate &&
            !value.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {
    }
}

static void gds_atomic_min(std::atomic<uint64_t> & value, uint64_t candidate) {
    uint64_t current = value.load(std::memory_order_relaxed);
    while (current > candidate &&
            !value.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {
    }
}

static void gds_atomic_gcd(std::atomic<uint64_t> & value, uint64_t candidate) {
    if (candidate == 0) {
        return;
    }
    uint64_t current = value.load(std::memory_order_relaxed);
    for (;;) {
        const uint64_t next = current == 0 ? candidate : std::gcd(current, candidate);
        if (value.compare_exchange_weak(current, next, std::memory_order_relaxed)) {
            return;
        }
    }
}

static void gds_record_read(DSLoader * loader, uint64_t bytes, uint64_t elapsed_us) {
    if (loader == nullptr) {
        return;
    }
    loader->work_read_us.fetch_add(elapsed_us, std::memory_order_relaxed);
    loader->work_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
    loader->work_read_calls.fetch_add(1, std::memory_order_relaxed);
    gds_atomic_max(loader->work_read_max_us, elapsed_us);
}

static void gds_record_batch_stats(
        DSLoader * loader,
        const DSLoaderStreamRequest * requests,
        int request_count,
        uint64_t wall_us,
        uint64_t read_sum_us,
        uint64_t read_max_us,
        bool attempted_batch_api,
        bool used_batch_api,
        bool used_worker_threads) {
    if (loader == nullptr || requests == nullptr || request_count <= 0) {
        return;
    }

    uint64_t bytes = 0;
    uint64_t aligned_4k = 0;
    uint64_t unaligned_4k = 0;
    uint64_t file_offset_4k_aligned = 0;
    uint64_t size_4k_aligned = 0;
    uint64_t cuda_dest_4k_aligned = 0;
    uint64_t file_offset_mod_min = UINT64_MAX;
    uint64_t file_offset_mod_max = 0;
    uint64_t file_offset_mod_gcd = 0;
    uint64_t size_mod_min = UINT64_MAX;
    uint64_t size_mod_max = 0;
    uint64_t size_mod_gcd = 0;
    uint64_t cuda_dest_mod_min = UINT64_MAX;
    uint64_t cuda_dest_mod_max = 0;
    uint64_t cuda_dest_mod_gcd = 0;
    uint64_t size_le_64k = 0;
    uint64_t size_le_256k = 0;
    uint64_t size_le_1m = 0;
    uint64_t size_le_4m = 0;
    uint64_t size_gt_4m = 0;
    uint64_t min_request = UINT64_MAX;
    uint64_t max_request = 0;

    for (int i = 0; i < request_count; ++i) {
        const uint64_t size = requests[i].size;
        bytes += size;
        min_request = std::min(min_request, size);
        max_request = std::max(max_request, size);
        if ((requests[i].file_offset % 4096) == 0 && (size % 4096) == 0) {
            aligned_4k++;
        } else {
            unaligned_4k++;
        }
        const uint64_t file_offset_mod = requests[i].file_offset % 4096;
        const uint64_t size_mod = size % 4096;
        const uint64_t cuda_dest_mod = requests[i].cuda_dest_ptr % 4096;
        file_offset_4k_aligned += file_offset_mod == 0 ? 1 : 0;
        size_4k_aligned += size_mod == 0 ? 1 : 0;
        cuda_dest_4k_aligned += cuda_dest_mod == 0 ? 1 : 0;
        if (file_offset_mod != 0) {
            file_offset_mod_min = std::min(file_offset_mod_min, file_offset_mod);
            file_offset_mod_max = std::max(file_offset_mod_max, file_offset_mod);
            file_offset_mod_gcd = file_offset_mod_gcd == 0
                    ? file_offset_mod
                    : std::gcd(file_offset_mod_gcd, file_offset_mod);
        }
        if (size_mod != 0) {
            size_mod_min = std::min(size_mod_min, size_mod);
            size_mod_max = std::max(size_mod_max, size_mod);
            size_mod_gcd = size_mod_gcd == 0 ? size_mod : std::gcd(size_mod_gcd, size_mod);
        }
        if (cuda_dest_mod != 0) {
            cuda_dest_mod_min = std::min(cuda_dest_mod_min, cuda_dest_mod);
            cuda_dest_mod_max = std::max(cuda_dest_mod_max, cuda_dest_mod);
            cuda_dest_mod_gcd = cuda_dest_mod_gcd == 0
                    ? cuda_dest_mod
                    : std::gcd(cuda_dest_mod_gcd, cuda_dest_mod);
        }

        if (size <= 64ULL * 1024ULL) {
            size_le_64k++;
        } else if (size <= 256ULL * 1024ULL) {
            size_le_256k++;
        } else if (size <= 1024ULL * 1024ULL) {
            size_le_1m++;
        } else if (size <= 4ULL * 1024ULL * 1024ULL) {
            size_le_4m++;
        } else {
            size_gt_4m++;
        }
    }

    loader->io_batches.fetch_add(1, std::memory_order_relaxed);
    loader->io_requests.fetch_add(uint64_t(request_count), std::memory_order_relaxed);
    loader->io_bytes.fetch_add(bytes, std::memory_order_relaxed);
    loader->io_wall_us.fetch_add(wall_us, std::memory_order_relaxed);
    loader->io_read_sum_us.fetch_add(read_sum_us, std::memory_order_relaxed);
    loader->io_batch_api_attempts.fetch_add(attempted_batch_api ? 1 : 0, std::memory_order_relaxed);
    loader->io_batch_api_successes.fetch_add(used_batch_api ? 1 : 0, std::memory_order_relaxed);
    loader->io_worker_batches.fetch_add(used_worker_threads ? 1 : 0, std::memory_order_relaxed);
    loader->io_serial_batches.fetch_add(!used_batch_api && !used_worker_threads ? 1 : 0, std::memory_order_relaxed);
    loader->io_aligned_4k_requests.fetch_add(aligned_4k, std::memory_order_relaxed);
    loader->io_unaligned_4k_requests.fetch_add(unaligned_4k, std::memory_order_relaxed);
    loader->io_file_offset_4k_aligned.fetch_add(file_offset_4k_aligned, std::memory_order_relaxed);
    loader->io_size_4k_aligned.fetch_add(size_4k_aligned, std::memory_order_relaxed);
    loader->io_cuda_dest_4k_aligned.fetch_add(cuda_dest_4k_aligned, std::memory_order_relaxed);
    if (file_offset_mod_min != UINT64_MAX) {
        gds_atomic_min(loader->io_file_offset_mod_min, file_offset_mod_min);
        gds_atomic_max(loader->io_file_offset_mod_max, file_offset_mod_max);
        gds_atomic_gcd(loader->io_file_offset_mod_gcd, file_offset_mod_gcd);
    }
    if (size_mod_min != UINT64_MAX) {
        gds_atomic_min(loader->io_size_mod_min, size_mod_min);
        gds_atomic_max(loader->io_size_mod_max, size_mod_max);
        gds_atomic_gcd(loader->io_size_mod_gcd, size_mod_gcd);
    }
    if (cuda_dest_mod_min != UINT64_MAX) {
        gds_atomic_min(loader->io_cuda_dest_mod_min, cuda_dest_mod_min);
        gds_atomic_max(loader->io_cuda_dest_mod_max, cuda_dest_mod_max);
        gds_atomic_gcd(loader->io_cuda_dest_mod_gcd, cuda_dest_mod_gcd);
    }
    loader->io_size_le_64k.fetch_add(size_le_64k, std::memory_order_relaxed);
    loader->io_size_le_256k.fetch_add(size_le_256k, std::memory_order_relaxed);
    loader->io_size_le_1m.fetch_add(size_le_1m, std::memory_order_relaxed);
    loader->io_size_le_4m.fetch_add(size_le_4m, std::memory_order_relaxed);
    loader->io_size_gt_4m.fetch_add(size_gt_4m, std::memory_order_relaxed);
    gds_atomic_min(loader->io_min_request_bytes, min_request);
    gds_atomic_max(loader->io_max_request_bytes, max_request);
    gds_atomic_max(loader->io_read_max_us, read_max_us);
    gds_atomic_max(loader->io_max_batch_requests, uint64_t(request_count));
    gds_atomic_max(loader->io_max_batch_bytes, bytes);
    gds_atomic_max(loader->io_max_batch_wall_us, wall_us);
}

static void close_file_handle(FileHandle & fh) {
    if (fh.cf_handle != nullptr && g_cuFileHandleDeregister != nullptr) {
        g_cuFileHandleDeregister(fh.cf_handle);
        fh.cf_handle = nullptr;
    }
    if (fh.fd >= 0) {
        close(fh.fd);
        fh.fd = -1;
    }
}

static FileHandle * get_file(DSLoader * loader, const wchar_t * file_path) {
    if (loader == nullptr || file_path == nullptr || !ensure_cufile_loaded()) {
        return nullptr;
    }

    const std::string path = narrow_path(file_path);
    if (path.empty()) {
        set_last_error(-EINVAL);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(loader->mutex);
    auto it = loader->files.find(path);
    if (it != loader->files.end()) {
        return &it->second;
    }

    FileHandle fh;
    fh.fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fh.fd < 0) {
        set_last_error(-errno);
        return nullptr;
    }

    CUfileDescr_t descr = {};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fh.fd;

    CUfileError_t status = g_cuFileHandleRegister(&fh.cf_handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        set_last_error(status.err != 0 ? status.err : status.cu_err);
        close_file_handle(fh);
        return nullptr;
    }

    auto inserted = loader->files.emplace(path, fh);
    return &inserted.first->second;
}

static FileHandle * get_thread_file(const wchar_t * file_path) {
    if (file_path == nullptr || !ensure_cufile_loaded()) {
        return nullptr;
    }

    struct ThreadFiles {
        std::unordered_map<std::string, FileHandle> files;

        ~ThreadFiles() {
            for (auto & entry : files) {
                close_file_handle(entry.second);
            }
            files.clear();
        }
    };

    const std::string path = narrow_path(file_path);
    if (path.empty()) {
        set_last_error(-EINVAL);
        return nullptr;
    }

    thread_local ThreadFiles thread_files;
    auto it = thread_files.files.find(path);
    if (it != thread_files.files.end()) {
        return &it->second;
    }

    FileHandle fh;
    fh.fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fh.fd < 0) {
        set_last_error(-errno);
        return nullptr;
    }

    CUfileDescr_t descr = {};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fh.fd;

    CUfileError_t status = g_cuFileHandleRegister(&fh.cf_handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        set_last_error(status.err != 0 ? status.err : status.cu_err);
        close_file_handle(fh);
        return nullptr;
    }

    auto inserted = thread_files.files.emplace(path, fh);
    return &inserted.first->second;
}

static int read_one_gds_request(DSLoader * loader, const DSLoaderStreamRequest & req) {
    if (req.uncompressed_size != 0 && req.uncompressed_size != req.size) {
        // The Linux GDS path streams raw GGUF bytes. GDeflate support would
        // need a separate GPU decompression path.
        set_last_error(-ENOTSUP);
        return -1;
    }

    // Plain buffered read + HtoD copy, used to A/B the cuFile/GDS path against an
    // ordinary pread that can benefit from the OS page cache. No cuFile, no O_DIRECT.
    if (gds_plain_read_enabled()) {
        if (!ensure_cuda_loaded()) {
            return -1;
        }
        const std::string path = narrow_path(req.file_path);
        if (path.empty()) {
            set_last_error(-EINVAL);
            return -1;
        }

        struct PlainFiles {
            std::unordered_map<std::string, int> fds;
            ~PlainFiles() {
                for (auto & e : fds) {
                    if (e.second >= 0) close(e.second);
                }
            }
        };
        thread_local PlainFiles plain_files;
        int fd = -1;
        auto fit = plain_files.fds.find(path);
        if (fit != plain_files.fds.end()) {
            fd = fit->second;
        } else {
            fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) {
                set_last_error(-errno);
                return -1;
            }
            plain_files.fds.emplace(path, fd);
        }

        struct PlainHostBuf {
            void * ptr = nullptr;
            uint64_t size = 0;
            ~PlainHostBuf() {
                if (ptr != nullptr && g_cuMemFreeHost != nullptr) {
                    g_cuMemFreeHost(ptr);
                }
            }
            bool ensure(uint64_t needed) {
                if (ptr != nullptr && size >= needed) {
                    return true;
                }
                if (ptr != nullptr) {
                    g_cuMemFreeHost(ptr);
                    ptr = nullptr;
                    size = 0;
                }
                void * next = nullptr;
                if (g_cuMemHostAlloc(&next, size_t(needed), 0) != CUDA_SUCCESS || next == nullptr) {
                    return false;
                }
                ptr = next;
                size = needed;
                return true;
            }
        };
        thread_local PlainHostBuf host_buf;
        if (!host_buf.ensure(req.size)) {
            set_last_error(-ENOMEM);
            return -1;
        }

        uint64_t done = 0;
        while (done < req.size) {
            const uint64_t read_t0 = gds_now_us();
            const ssize_t nread = pread(
                    fd,
                    static_cast<char *>(host_buf.ptr) + done,
                    size_t(req.size - done),
                    off_t(req.file_offset + done));
            if (nread > 0) {
                gds_record_read(loader, uint64_t(nread), gds_now_us() - read_t0);
                done += uint64_t(nread);
            } else {
                set_last_error(nread == 0 ? -EIO : -errno);
                return -1;
            }
        }

        const CUresult cr = g_cuMemcpyHtoD(
                CUdeviceptr(req.cuda_dest_ptr), host_buf.ptr, size_t(req.size));
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return -1;
        }
        return 0;
    }

    FileHandle * fh = gds_thread_file_handles_enabled()
            ? get_thread_file(req.file_path)
            : get_file(loader, req.file_path);
    if (fh == nullptr || fh->cf_handle == nullptr) {
        return -1;
    }

    if (gds_aligned_staging_enabled() &&
            ((req.file_offset % 4096) != 0 || (req.size % 4096) != 0)) {
        if (!ensure_cuda_loaded()) {
            return -1;
        }
        const uint64_t aligned_file_offset = req.file_offset & ~4095ULL;
        const uint64_t prefix = req.file_offset - aligned_file_offset;
        const uint64_t aligned_size = (prefix + req.size + 4095ULL) & ~4095ULL;

        const CUresult ctx_result = g_cuCtxSetCurrent(g_cuda_context);
        if (ctx_result != CUDA_SUCCESS) {
            set_last_error(ctx_result);
            return -1;
        }

        CUdeviceptr staging = 0;
        CUresult cr = g_cuMemAlloc(&staging, size_t(aligned_size));
        if (cr != CUDA_SUCCESS || staging == 0) {
            set_last_error(cr);
            return -1;
        }

        void * staging_ptr = reinterpret_cast<void *>(uintptr_t(staging));
        bool staging_registered = false;
        const CUfileError_t reg_status = g_cuFileBufRegister(staging_ptr, size_t(aligned_size), 0);
        staging_registered = reg_status.err == CU_FILE_SUCCESS;

        uint64_t done = 0;
        while (done < aligned_size) {
            const size_t chunk = size_t(std::min<uint64_t>(
                    aligned_size - done,
                    1024ull * 1024ull * 1024ull));
            const uint64_t read_t0 = gds_now_us();
            const ssize_t nread = g_cuFileRead(
                    fh->cf_handle,
                    staging_ptr,
                    chunk,
                    off_t(aligned_file_offset + done),
                    off_t(done));
            if (nread > 0) {
                gds_record_read(loader, uint64_t(nread), gds_now_us() - read_t0);
            }
            if (nread <= 0) {
                set_last_error(nread == 0 ? -EIO : int32_t(nread));
                if (staging_registered) {
                    g_cuFileBufDeregister(staging_ptr);
                }
                g_cuMemFree(staging);
                return -1;
            }
            done += uint64_t(nread);
        }

        cr = g_cuMemcpyDtoD(
                CUdeviceptr(req.cuda_dest_ptr),
                staging + prefix,
                size_t(req.size));
        if (staging_registered) {
            g_cuFileBufDeregister(staging_ptr);
        }
        g_cuMemFree(staging);
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return -1;
        }
        return 0;
    }

    if (gds_stage_unaligned_dest_enabled() &&
            (req.cuda_dest_ptr % 4096) != 0 &&
            (req.file_offset % 4096) == 0 &&
            (req.size % 4096) == 0) {
        if (!ensure_cuda_loaded()) {
            return -1;
        }

        struct ThreadStaging {
            CUdeviceptr raw = 0;
            CUdeviceptr visible = 0;
            uint64_t size = 0;
            bool registered = false;

            ~ThreadStaging() {
                if (registered && g_cuFileBufDeregister != nullptr) {
                    g_cuFileBufDeregister(reinterpret_cast<void *>(uintptr_t(visible)));
                }
                if (raw != 0 && g_cuMemFree != nullptr) {
                    g_cuMemFree(raw);
                }
            }

            bool ensure(uint64_t needed) {
                if (visible != 0 && size >= needed) {
                    return true;
                }
                if (registered && g_cuFileBufDeregister != nullptr) {
                    g_cuFileBufDeregister(reinterpret_cast<void *>(uintptr_t(visible)));
                    registered = false;
                }
                if (raw != 0 && g_cuMemFree != nullptr) {
                    g_cuMemFree(raw);
                    raw = 0;
                    visible = 0;
                    size = 0;
                }
                const uint64_t alloc_size = needed + 4095ULL;
                CUdeviceptr next_raw = 0;
                const CUresult cr = g_cuMemAlloc(&next_raw, size_t(alloc_size));
                if (cr != CUDA_SUCCESS || next_raw == 0) {
                    set_last_error(cr);
                    return false;
                }
                const CUdeviceptr next_visible = (next_raw + 4095ULL) & ~4095ULL;
                const CUfileError_t reg_status = g_cuFileBufRegister(
                        reinterpret_cast<void *>(uintptr_t(next_visible)),
                        size_t(needed),
                        0);
                raw = next_raw;
                visible = next_visible;
                size = needed;
                registered = reg_status.err == CU_FILE_SUCCESS;
                return true;
            }
        };

        thread_local ThreadStaging staging;
        const CUresult ctx_result = g_cuCtxSetCurrent(g_cuda_context);
        if (ctx_result != CUDA_SUCCESS) {
            set_last_error(ctx_result);
            return -1;
        }
        if (!staging.ensure(req.size)) {
            return -1;
        }

        uint64_t done = 0;
        while (done < req.size) {
            const size_t chunk = size_t(std::min<uint64_t>(
                    req.size - done,
                    1024ull * 1024ull * 1024ull));
            const uint64_t read_t0 = gds_now_us();
            const ssize_t nread = g_cuFileRead(
                    fh->cf_handle,
                    reinterpret_cast<void *>(uintptr_t(staging.visible)),
                    chunk,
                    off_t(req.file_offset + done),
                    off_t(done));
            if (nread > 0) {
                gds_record_read(loader, uint64_t(nread), gds_now_us() - read_t0);
            }
            if (nread <= 0) {
                set_last_error(nread == 0 ? -EIO : int32_t(nread));
                return -1;
            }
            done += uint64_t(nread);
        }

        const CUresult cr = g_cuMemcpyDtoD(
                CUdeviceptr(req.cuda_dest_ptr),
                staging.visible,
                size_t(req.size));
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return -1;
        }
        return 0;
    }

    RegisteredCudaBuffer registered_buffer;
    const bool already_registered =
            find_registered_cuda_buffer(req.cuda_dest_ptr, req.size, registered_buffer);
    void * dst = reinterpret_cast<void *>(uintptr_t(req.cuda_dest_ptr));
    void * io_base = already_registered
            ? reinterpret_cast<void *>(uintptr_t(registered_buffer.ptr))
            : dst;
    const uint64_t io_offset = already_registered
            ? req.cuda_dest_ptr - registered_buffer.ptr
            : 0;
    bool registered_for_request = false;
    if (!already_registered && gds_register_per_request()) {
        const CUfileError_t reg_status = g_cuFileBufRegister(dst, req.size, 0);
        registered_for_request = reg_status.err == CU_FILE_SUCCESS;
    }

    uint64_t done = 0;
    while (done < req.size) {
        const size_t chunk = size_t(std::min<uint64_t>(req.size - done, 1024ull * 1024ull * 1024ull));
        const uint64_t read_t0 = gds_now_us();
        const ssize_t nread = g_cuFileRead(
                fh->cf_handle,
                io_base,
                chunk,
                off_t(req.file_offset + done),
                off_t(io_offset + done));
        if (nread > 0) {
            gds_record_read(loader, uint64_t(nread), gds_now_us() - read_t0);
        }
        if (nread <= 0) {
            set_last_error(nread == 0 ? -EIO : int32_t(nread));
            if (registered_for_request) {
                g_cuFileBufDeregister(dst);
            }
            return -1;
        }
        done += uint64_t(nread);
    }

    if (registered_for_request) {
        g_cuFileBufDeregister(dst);
    }

    return 0;
}

static void gds_worker_loop(DSLoader * loader) {
    uint64_t seen_generation = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(loader->work_mutex);
            loader->work_cv.wait(lock, [&] {
                return loader->work_stop || loader->work_generation != seen_generation;
            });
            if (loader->work_stop) {
                return;
            }
            seen_generation = loader->work_generation;
        }

        for (;;) {
            const int claim_count = gds_worker_claim_count();
            const int begin = loader->work_next_index.fetch_add(claim_count, std::memory_order_relaxed);
            if (begin >= loader->work_request_count) {
                break;
            }
            const int end = std::min(begin + claim_count, loader->work_request_count);
            for (int index = begin; index < end; ++index) {
                if (loader->work_failed.load(std::memory_order_relaxed) == 0 &&
                        read_one_gds_request(loader, loader->work_requests[index]) != 0) {
                    loader->work_failed.store(1, std::memory_order_relaxed);
                }
            }
            if (loader->work_remaining.fetch_sub(end - begin, std::memory_order_acq_rel) == end - begin) {
                std::lock_guard<std::mutex> lock(loader->work_mutex);
                loader->work_completed_generation = seen_generation;
                loader->done_cv.notify_one();
            }
        }
    }
}

static void ensure_read_workers(DSLoader * loader) {
    if (loader == nullptr || !loader->read_workers.empty()) {
        return;
    }

    const int n_threads = gds_read_threads();
    if (n_threads <= 1) {
        return;
    }

    loader->read_workers.reserve(size_t(n_threads));
    for (int i = 0; i < n_threads; ++i) {
        loader->read_workers.emplace_back(gds_worker_loop, loader);
    }
}

static void stop_read_workers(DSLoader * loader) {
    if (loader == nullptr || loader->read_workers.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(loader->work_mutex);
        loader->work_stop = true;
    }
    loader->work_cv.notify_all();
    for (std::thread & worker : loader->read_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    loader->read_workers.clear();
}

static int read_gds_batch(
        DSLoader * loader,
        const DSLoaderStreamRequest * requests,
        int request_count) {
    if (!gds_use_batch() ||
            g_cuFileBatchIOSetUp == nullptr ||
            g_cuFileBatchIOSubmit == nullptr ||
            g_cuFileBatchIOGetStatus == nullptr ||
            g_cuFileBatchIODestroy == nullptr ||
            request_count <= 1) {
        return 1;
    }

    std::vector<CUfileIOParams_t> params(static_cast<size_t>(request_count));
    std::vector<size_t> expected_sizes(static_cast<size_t>(request_count));
    for (int i = 0; i < request_count; ++i) {
        const DSLoaderStreamRequest & req = requests[i];
        if ((req.uncompressed_size != 0 && req.uncompressed_size != req.size) ||
                req.size > size_t(-1)) {
            return 1;
        }

        FileHandle * fh = get_file(loader, req.file_path);
        if (fh == nullptr || fh->cf_handle == nullptr) {
            return -1;
        }

        RegisteredCudaBuffer registered_buffer;
        const bool registered =
                find_registered_cuda_buffer(req.cuda_dest_ptr, req.size, registered_buffer);
        CUfileIOParams_t & param = params[size_t(i)];
        param.mode = CUFILE_BATCH;
        param.u.batch.devPtr_base = reinterpret_cast<void *>(uintptr_t(
                registered ? registered_buffer.ptr : req.cuda_dest_ptr));
        param.u.batch.file_offset = off_t(req.file_offset);
        param.u.batch.devPtr_offset = off_t(
                registered ? req.cuda_dest_ptr - registered_buffer.ptr : 0);
        param.u.batch.size = size_t(req.size);
        param.fh = fh->cf_handle;
        param.opcode = CUFILE_READ;
        param.cookie = reinterpret_cast<void *>(uintptr_t(i + 1));
        expected_sizes[size_t(i)] = size_t(req.size);
    }

    CUfileBatchHandle_t batch = nullptr;
    CUfileError_t status = g_cuFileBatchIOSetUp(&batch, unsigned(request_count));
    if (status.err != CU_FILE_SUCCESS || batch == nullptr) {
        return 1;
    }

    status = g_cuFileBatchIOSubmit(batch, unsigned(request_count), params.data(), 0);
    if (status.err != CU_FILE_SUCCESS) {
        g_cuFileBatchIODestroy(batch);
        return 1;
    }

    int completed = 0;
    std::vector<CUfileIOEvents_t> events(static_cast<size_t>(request_count));
    while (completed < request_count) {
        unsigned nr = unsigned(request_count - completed);
        status = g_cuFileBatchIOGetStatus(batch, 1, &nr, events.data(), nullptr);
        if (status.err != CU_FILE_SUCCESS || nr == 0) {
            set_last_error(status.err != CU_FILE_SUCCESS ? status.err : -EIO);
            if (g_cuFileBatchIOCancel != nullptr) {
                g_cuFileBatchIOCancel(batch);
            }
            g_cuFileBatchIODestroy(batch);
            return -1;
        }

        for (unsigned i = 0; i < nr; ++i) {
            const CUfileIOEvents_t & event = events[i];
            const uintptr_t cookie = uintptr_t(event.cookie);
            if (cookie == 0 || cookie > size_t(request_count) ||
                    event.status != CUFILE_COMPLETE ||
                    event.ret != expected_sizes[cookie - 1]) {
                set_last_error(-EIO);
                if (g_cuFileBatchIOCancel != nullptr) {
                    g_cuFileBatchIOCancel(batch);
                }
                g_cuFileBatchIODestroy(batch);
                return -1;
            }
        }
        completed += int(nr);
    }

    g_cuFileBatchIODestroy(batch);
    return 0;
}

int ds_loader_available() {
    return ensure_cufile_loaded() ? 1 : 0;
}

int32_t ds_loader_get_hresult() {
    return g_last_error;
}

DSLoaderHandle ds_loader_create() {
    if (!ensure_cufile_loaded()) {
        return nullptr;
    }
    DSLoader * loader = new DSLoader();
    if (g_cuStreamCreate != nullptr &&
            g_cuStreamSynchronize != nullptr &&
            g_cuStreamDestroy != nullptr) {
        g_cuStreamCreate(&loader->h2d_stream, 1);
        g_cuStreamCreate(&loader->d2h_stream, 1);
    }
    ensure_read_workers(loader);
    return loader;
}

void ds_loader_destroy(DSLoaderHandle loader) {
    if (loader == nullptr) {
        return;
    }
    stop_read_workers(loader);
    if (ensure_cuda_loaded() && g_cuStreamDestroy != nullptr) {
        if (loader->h2d_stream != nullptr) {
            g_cuStreamDestroy(loader->h2d_stream);
            loader->h2d_stream = nullptr;
        }
        if (loader->d2h_stream != nullptr) {
            g_cuStreamDestroy(loader->d2h_stream);
            loader->d2h_stream = nullptr;
        }
    }
    {
        std::lock_guard<std::mutex> lock(loader->mutex);
        for (auto & entry : loader->files) {
            close_file_handle(entry.second);
        }
        loader->files.clear();
    }
    delete loader;
}

void * ds_loader_create_gpu_buffer(DSLoaderHandle, uint64_t) {
    set_last_error(-ENOSYS);
    return nullptr;
}

void ds_loader_destroy_gpu_buffer(void *) {
}

int ds_loader_read(DSLoaderHandle, const wchar_t *, uint64_t, uint64_t, void *) {
    set_last_error(-ENOSYS);
    return -1;
}

int ds_loader_read_chunked(DSLoaderHandle, const wchar_t *, uint64_t, uint64_t, void *) {
    set_last_error(-ENOSYS);
    return -1;
}

int ds_loader_read_to_memory(DSLoaderHandle, const wchar_t *, uint64_t, uint64_t, void *) {
    set_last_error(-ENOSYS);
    return -1;
}

int ds_loader_open_file(DSLoaderHandle loader, const wchar_t * file_path) {
    return get_file(loader, file_path) != nullptr ? 0 : -1;
}

void ds_loader_close_file(DSLoaderHandle loader) {
    if (loader == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(loader->mutex);
    for (auto & entry : loader->files) {
        close_file_handle(entry.second);
    }
    loader->files.clear();
}

int ds_loader_enqueue_read(DSLoaderHandle, uint64_t, uint64_t, void *, uint64_t) {
    set_last_error(-ENOSYS);
    return -1;
}

int ds_loader_submit_and_wait(DSLoaderHandle) {
    return 0;
}

int ds_loader_submit(DSLoaderHandle) {
    return 0;
}

int ds_loader_is_complete(DSLoaderHandle) {
    return 1;
}

int ds_loader_wait_complete(DSLoaderHandle) {
    return 0;
}

int ds_loader_gpu_readback(DSLoaderHandle, void *, uint64_t, void *) {
    set_last_error(-ENOSYS);
    return -1;
}

int ds_loader_debug_shared(DSLoaderHandle) {
    return ds_loader_available();
}

int ds_loader_cuda_available() {
    return ensure_cuda_loaded() ? 1 : 0;
}

void * ds_loader_create_shared_gpu_buffer(DSLoaderHandle, uint64_t) {
    set_last_error(-ENOSYS);
    return nullptr;
}

CUDAInteropHandle ds_loader_export_to_cuda(DSLoaderHandle, void *, uint64_t) {
    set_last_error(-ENOSYS);
    return nullptr;
}

uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle) {
    return 0;
}

int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle, void *, uint64_t) {
    set_last_error(-ENOSYS);
    return -1;
}

void ds_loader_cuda_destroy(CUDAInteropHandle) {
}

int ds_loader_stream_to_cuda(
        DSLoaderHandle loader,
        const wchar_t * file_path,
        uint64_t file_offset,
        uint64_t size,
        uint64_t cuda_dest_ptr) {
    DSLoaderStreamRequest req = {};
    req.file_path = file_path;
    req.file_offset = file_offset;
    req.size = size;
    req.cuda_dest_ptr = cuda_dest_ptr;
    req.uncompressed_size = size;
    return ds_loader_stream_to_cuda_batch(loader, &req, 1);
}

int ds_loader_stream_to_cuda_batch(
        DSLoaderHandle loader,
        const DSLoaderStreamRequest * requests,
        int request_count) {
    if (loader == nullptr || requests == nullptr || request_count < 0 || !ensure_cufile_loaded()) {
        set_last_error(-EINVAL);
        return -1;
    }

    const bool profile = gds_io_profile_enabled();
    const uint64_t batch_t0 = gds_now_us();
    loader->work_read_us.store(0, std::memory_order_relaxed);
    loader->work_read_bytes.store(0, std::memory_order_relaxed);
    loader->work_read_calls.store(0, std::memory_order_relaxed);
    loader->work_read_max_us.store(0, std::memory_order_relaxed);

    const bool attempted_batch_api =
            gds_use_batch() &&
            g_cuFileBatchIOSetUp != nullptr &&
            g_cuFileBatchIOSubmit != nullptr &&
            g_cuFileBatchIOGetStatus != nullptr &&
            g_cuFileBatchIODestroy != nullptr &&
            request_count > 1;
    const int batch_result = read_gds_batch(loader, requests, request_count);
    if (batch_result <= 0) {
        if (batch_result == 0) {
            const uint64_t wall_us = gds_now_us() - batch_t0;
            uint64_t bytes = 0;
            for (int i = 0; i < request_count; ++i) {
                bytes += requests[i].size;
            }
            gds_record_batch_stats(
                    loader, requests, request_count,
                    wall_us, wall_us, wall_us,
                    attempted_batch_api, true, false);
            if (profile) {
                std::fprintf(stderr,
                        "GDS_IO_PROFILE mode=batch_api requests=%d reads=%d bytes=%" PRIu64
                        " wall_us=%" PRIu64 " io_sum_us=%" PRIu64 " io_max_us=%" PRIu64
                        " effective_concurrency=%.2f\n",
                        request_count,
                        request_count,
                        bytes,
                        wall_us,
                        wall_us,
                        wall_us,
                        1.0);
            }
        }
        return batch_result;
    }

    const int n_threads = std::min<int>(int(loader->read_workers.size()), std::max(1, request_count));
    if (n_threads <= 1 || request_count <= 1 || loader->read_workers.empty()) {
        for (int i = 0; i < request_count; ++i) {
            if (read_one_gds_request(loader, requests[i]) != 0) {
                return -1;
            }
        }
        const uint64_t wall_us = gds_now_us() - batch_t0;
        const uint64_t io_us = loader->work_read_us.load(std::memory_order_relaxed);
        gds_record_batch_stats(
                loader, requests, request_count,
                wall_us, io_us,
                loader->work_read_max_us.load(std::memory_order_relaxed),
                attempted_batch_api, false, false);
        if (profile) {
            std::fprintf(stderr,
                    "GDS_IO_PROFILE mode=serial requests=%d reads=%" PRIu64 " bytes=%" PRIu64
                    " wall_us=%" PRIu64 " io_sum_us=%" PRIu64 " io_max_us=%" PRIu64
                    " effective_concurrency=%.2f\n",
                    request_count,
                    loader->work_read_calls.load(std::memory_order_relaxed),
                    loader->work_read_bytes.load(std::memory_order_relaxed),
                    wall_us,
                    io_us,
                    loader->work_read_max_us.load(std::memory_order_relaxed),
                    wall_us > 0 ? double(io_us) / double(wall_us) : 0.0);
        }
        return 0;
    }

    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(loader->work_mutex);
        loader->work_requests = requests;
        loader->work_request_count = request_count;
        loader->work_next_index.store(0, std::memory_order_relaxed);
        loader->work_remaining.store(request_count, std::memory_order_relaxed);
        loader->work_failed.store(0, std::memory_order_relaxed);
        generation = ++loader->work_generation;
    }
    loader->work_cv.notify_all();

    {
        std::unique_lock<std::mutex> lock(loader->work_mutex);
        loader->done_cv.wait(lock, [&] {
            return loader->work_completed_generation == generation;
        });
    }

    if (loader->work_failed.load(std::memory_order_relaxed) != 0) {
        return -1;
    }

    const uint64_t wall_us = gds_now_us() - batch_t0;
    const uint64_t io_us = loader->work_read_us.load(std::memory_order_relaxed);
    gds_record_batch_stats(
            loader, requests, request_count,
            wall_us, io_us,
            loader->work_read_max_us.load(std::memory_order_relaxed),
            attempted_batch_api, false, true);
    if (profile) {
        std::fprintf(stderr,
                "GDS_IO_PROFILE mode=workers threads=%d requests=%d reads=%" PRIu64 " bytes=%" PRIu64
                " wall_us=%" PRIu64 " io_sum_us=%" PRIu64 " io_max_us=%" PRIu64
                " effective_concurrency=%.2f\n",
                n_threads,
                request_count,
                loader->work_read_calls.load(std::memory_order_relaxed),
                loader->work_read_bytes.load(std::memory_order_relaxed),
                wall_us,
                io_us,
                loader->work_read_max_us.load(std::memory_order_relaxed),
                wall_us > 0 ? double(io_us) / double(wall_us) : 0.0);
    }

    return 0;
}

int ds_loader_get_io_stats(DSLoaderHandle loader, DSLoaderIoStats * out_stats) {
    if (loader == nullptr || out_stats == nullptr) {
        set_last_error(-EINVAL);
        return -1;
    }

    DSLoaderIoStats stats = {};
    stats.batches = loader->io_batches.load(std::memory_order_relaxed);
    stats.requests = loader->io_requests.load(std::memory_order_relaxed);
    stats.bytes = loader->io_bytes.load(std::memory_order_relaxed);
    stats.wall_us = loader->io_wall_us.load(std::memory_order_relaxed);
    stats.read_sum_us = loader->io_read_sum_us.load(std::memory_order_relaxed);
    stats.read_max_us = loader->io_read_max_us.load(std::memory_order_relaxed);
    stats.batch_api_attempts = loader->io_batch_api_attempts.load(std::memory_order_relaxed);
    stats.batch_api_successes = loader->io_batch_api_successes.load(std::memory_order_relaxed);
    stats.worker_batches = loader->io_worker_batches.load(std::memory_order_relaxed);
    stats.serial_batches = loader->io_serial_batches.load(std::memory_order_relaxed);
    stats.aligned_4k_requests = loader->io_aligned_4k_requests.load(std::memory_order_relaxed);
    stats.unaligned_4k_requests = loader->io_unaligned_4k_requests.load(std::memory_order_relaxed);
    stats.file_offset_4k_aligned = loader->io_file_offset_4k_aligned.load(std::memory_order_relaxed);
    stats.size_4k_aligned = loader->io_size_4k_aligned.load(std::memory_order_relaxed);
    stats.cuda_dest_4k_aligned = loader->io_cuda_dest_4k_aligned.load(std::memory_order_relaxed);
    stats.file_offset_mod_min = loader->io_file_offset_mod_min.load(std::memory_order_relaxed);
    if (stats.file_offset_mod_min == UINT64_MAX) {
        stats.file_offset_mod_min = 0;
    }
    stats.file_offset_mod_max = loader->io_file_offset_mod_max.load(std::memory_order_relaxed);
    stats.file_offset_mod_gcd = loader->io_file_offset_mod_gcd.load(std::memory_order_relaxed);
    stats.size_mod_min = loader->io_size_mod_min.load(std::memory_order_relaxed);
    if (stats.size_mod_min == UINT64_MAX) {
        stats.size_mod_min = 0;
    }
    stats.size_mod_max = loader->io_size_mod_max.load(std::memory_order_relaxed);
    stats.size_mod_gcd = loader->io_size_mod_gcd.load(std::memory_order_relaxed);
    stats.cuda_dest_mod_min = loader->io_cuda_dest_mod_min.load(std::memory_order_relaxed);
    if (stats.cuda_dest_mod_min == UINT64_MAX) {
        stats.cuda_dest_mod_min = 0;
    }
    stats.cuda_dest_mod_max = loader->io_cuda_dest_mod_max.load(std::memory_order_relaxed);
    stats.cuda_dest_mod_gcd = loader->io_cuda_dest_mod_gcd.load(std::memory_order_relaxed);
    stats.size_le_64k = loader->io_size_le_64k.load(std::memory_order_relaxed);
    stats.size_le_256k = loader->io_size_le_256k.load(std::memory_order_relaxed);
    stats.size_le_1m = loader->io_size_le_1m.load(std::memory_order_relaxed);
    stats.size_le_4m = loader->io_size_le_4m.load(std::memory_order_relaxed);
    stats.size_gt_4m = loader->io_size_gt_4m.load(std::memory_order_relaxed);
    stats.min_request_bytes = loader->io_min_request_bytes.load(std::memory_order_relaxed);
    if (stats.min_request_bytes == UINT64_MAX) {
        stats.min_request_bytes = 0;
    }
    stats.max_request_bytes = loader->io_max_request_bytes.load(std::memory_order_relaxed);
    stats.max_batch_requests = loader->io_max_batch_requests.load(std::memory_order_relaxed);
    stats.max_batch_bytes = loader->io_max_batch_bytes.load(std::memory_order_relaxed);
    stats.max_batch_wall_us = loader->io_max_batch_wall_us.load(std::memory_order_relaxed);
    *out_stats = stats;
    return 0;
}

void * ds_loader_host_alloc(uint64_t size, int * out_is_pinned) {
    if (out_is_pinned != nullptr) {
        *out_is_pinned = 0;
    }
    if (ensure_cuda_loaded()) {
        void * ptr = nullptr;
        if (g_cuMemHostAlloc(&ptr, size_t(size), 0) == CUDA_SUCCESS && ptr != nullptr) {
            if (out_is_pinned != nullptr) {
                *out_is_pinned = 1;
            }
            return ptr;
        }
    }

    void * ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size_t(size)) != 0) {
        return nullptr;
    }
    return ptr;
}

void ds_loader_host_free(void * ptr, int is_pinned) {
    if (ptr == nullptr) {
        return;
    }
    if (is_pinned && ensure_cuda_loaded()) {
        g_cuMemFreeHost(ptr);
        return;
    }
    free(ptr);
}

int ds_loader_host_to_cuda_batch(
        DSLoaderHandle loader,
        const DSLoaderHostToCudaRequest * requests,
        int request_count) {
    if (loader == nullptr || requests == nullptr || request_count < 0 || !ensure_cuda_loaded()) {
        set_last_error(-EINVAL);
        return -1;
    }
    std::lock_guard<std::mutex> lock(loader->h2d_mutex);
    const bool use_async =
            loader->h2d_stream != nullptr &&
            g_cuMemcpyHtoDAsync != nullptr &&
            g_cuStreamSynchronize != nullptr;
    for (int i = 0; i < request_count; ++i) {
        const DSLoaderHostToCudaRequest & req = requests[i];
        const CUresult cr = use_async
                ? g_cuMemcpyHtoDAsync(
                        CUdeviceptr(req.cuda_dest_ptr),
                        req.host_src_ptr,
                        size_t(req.size),
                        loader->h2d_stream)
                : g_cuMemcpyHtoD(
                        CUdeviceptr(req.cuda_dest_ptr),
                        req.host_src_ptr,
                        size_t(req.size));
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return -1;
        }
    }
    if (use_async) {
        const CUresult cr = g_cuStreamSynchronize(loader->h2d_stream);
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return -1;
        }
    }
    return 0;
}

int ds_loader_cuda_to_host_batch(
        DSLoaderHandle loader,
        const DSLoaderCudaToHostRequest * requests,
        int request_count) {
    if (loader == nullptr || requests == nullptr || request_count < 0 || !ensure_cuda_loaded()) {
        set_last_error(-EINVAL);
        return -1;
    }
    std::lock_guard<std::mutex> lock(loader->d2h_mutex);
    const bool use_async =
            loader->d2h_stream != nullptr &&
            g_cuMemcpyDtoHAsync != nullptr &&
            g_cuStreamSynchronize != nullptr;
    for (int i = 0; i < request_count; ++i) {
        const DSLoaderCudaToHostRequest & req = requests[i];
        const CUresult cr = use_async
                ? g_cuMemcpyDtoHAsync(
                        req.host_dest_ptr,
                        CUdeviceptr(req.cuda_src_ptr),
                        size_t(req.size),
                        loader->d2h_stream)
                : g_cuMemcpyDtoH(
                        req.host_dest_ptr,
                        CUdeviceptr(req.cuda_src_ptr),
                        size_t(req.size));
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return -1;
        }
    }
    if (use_async) {
        const CUresult cr = g_cuStreamSynchronize(loader->d2h_stream);
        if (cr != CUDA_SUCCESS) {
            set_last_error(cr);
            return -1;
        }
    }
    return 0;
}

uint64_t ds_loader_cuda_alloc(uint64_t size) {
    if (!ensure_cuda_loaded()) {
        return 0;
    }
    CUdeviceptr ptr = 0;
    const CUresult cr = g_cuMemAlloc(&ptr, size_t(size));
    if (cr != CUDA_SUCCESS) {
        set_last_error(cr);
        return 0;
    }
    if (ensure_cufile_loaded()) {
        void * raw_ptr = reinterpret_cast<void *>(uintptr_t(ptr));
        const CUfileError_t status = g_cuFileBufRegister(raw_ptr, size_t(size), 0);
        if (status.err == CU_FILE_SUCCESS) {
            remember_registered_cuda_buffer(uint64_t(ptr), size);
        } else {
            // Keep the allocation usable. cuFile can still use its internal
            // compatibility buffers, just with more overhead for small reads.
            set_last_error(status.err != 0 ? status.err : status.cu_err);
        }
    }
    return uint64_t(ptr);
}

void ds_loader_cuda_free(uint64_t ptr) {
    if (ptr != 0 && ensure_cuda_loaded()) {
        if (ensure_cufile_loaded() && forget_registered_cuda_buffer(ptr)) {
            g_cuFileBufDeregister(reinterpret_cast<void *>(uintptr_t(ptr)));
        }
        g_cuMemFree(CUdeviceptr(ptr));
    }
}

int ds_loader_cuda_wait_event(DSLoaderHandle) {
    if (!ensure_cuda_loaded()) {
        return -1;
    }
    return 0;
}

int ds_loader_cuda_dtoh(uint64_t src_cuda_ptr, void * dest, uint64_t size) {
    if (dest == nullptr || !ensure_cuda_loaded()) {
        set_last_error(-EINVAL);
        return -1;
    }
    const CUresult cr = g_cuMemcpyDtoH(dest, CUdeviceptr(src_cuda_ptr), size_t(size));
    if (cr != CUDA_SUCCESS) {
        set_last_error(cr);
        return -1;
    }
    return 0;
}

int ds_loader_cuda_dtod(uint64_t dst_cuda_ptr, uint64_t src_cuda_ptr, uint64_t size) {
    if (!ensure_cuda_loaded()) {
        return -1;
    }
    const CUresult cr = g_cuMemcpyDtoD(CUdeviceptr(dst_cuda_ptr), CUdeviceptr(src_cuda_ptr), size_t(size));
    if (cr != CUDA_SUCCESS) {
        set_last_error(cr);
        return -1;
    }
    return 0;
}

int ds_loader_compress_buffer(
        const void *,
        uint64_t,
        void *,
        uint64_t,
        uint64_t *) {
    set_last_error(-ENOSYS);
    return -1;
}

#endif // defined(__linux__)
