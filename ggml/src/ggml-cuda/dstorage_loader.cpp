// DirectStorage loader implementation
// Windows-only: Requires Windows 11 and DirectStorage 1.2+

#ifdef _WIN32

#include "dstorage_loader.h"
#include <windows.h>
#include <dstorage.h>
#include <dstorageerr.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "ole32.lib")

using Microsoft::WRL::ComPtr;

// Dynamically load DStorageGetFactory from dstorage.dll
typedef HRESULT (WINAPI *PFN_DStorageGetFactory)(REFIID riid, void** ppv);
typedef HRESULT (WINAPI *PFN_DStorageCreateCompressionCodec)(DSTORAGE_COMPRESSION_FORMAT format, UINT32 numThreads, REFIID riid, void** ppv);

static PFN_DStorageGetFactory g_DStorageGetFactory = nullptr;
static PFN_DStorageCreateCompressionCodec g_DStorageCreateCompressionCodec = nullptr;
static HMODULE g_dstorageModule = nullptr;
static HMODULE g_dstorageCoreModule = nullptr;

// ============================================================
// CUDA Driver API types and function pointers
// Defined manually — no CUDA SDK/headers needed.
// nvcuda.dll ships with every NVIDIA display driver.
// ============================================================

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef void* CUexternalMemory;

#define CUDA_SUCCESS 0
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1

typedef enum {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP     = 4,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
} CUexternalMemoryHandleType;

// Must match CUDA driver API struct layout exactly (x64, default packing)
struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 {
    CUexternalMemoryHandleType type;    // 4 bytes + 4 padding (union aligned to 8)
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;                           // 16 bytes (two pointers)
    unsigned long long size;            // 8 bytes
    unsigned int flags;                 // 4 bytes
    unsigned int reserved[16];          // 64 bytes
};

struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 {
    unsigned long long offset;          // 8 bytes
    unsigned long long size;            // 8 bytes
    unsigned int flags;                 // 4 bytes
    unsigned int reserved[16];          // 64 bytes
};

// CUDA driver API function pointer types
typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*PFN_cuImportExternalMemory)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1*);
typedef CUresult (*PFN_cuExternalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1*);
typedef CUresult (*PFN_cuDestroyExternalMemory)(CUexternalMemory);
typedef CUresult (*PFN_cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemcpyDtoD_v2)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t);
typedef CUresult (*PFN_cuMemAlloc_v2)(CUdeviceptr*, size_t);
typedef CUresult (*PFN_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*PFN_cuMemHostAlloc_v2)(void**, size_t, unsigned int);
typedef CUresult (*PFN_cuMemFreeHost)(void*);
typedef CUresult (*PFN_cuDeviceGetLuid)(char* luid, unsigned int* deviceNodeMask, CUdevice dev);
typedef CUresult (*PFN_cuCtxSynchronize)(void);

typedef CUresult (*PFN_cuMemcpyDtoDAsync_v2)(CUdeviceptr, CUdeviceptr, size_t, void*);
typedef CUresult (*PFN_cuMemcpyHtoDAsync_v2)(CUdeviceptr, const void*, size_t, void*);
typedef CUresult (*PFN_cuStreamCreate)(void**, unsigned int);
typedef CUresult (*PFN_cuStreamDestroy)(void*);
typedef CUresult (*PFN_cuEventCreate)(void**, unsigned int);
typedef CUresult (*PFN_cuEventRecord)(void*, void*);
typedef CUresult (*PFN_cuEventSynchronize)(void*);
typedef CUresult (*PFN_cuEventDestroy)(void*);

static HMODULE g_nvcudaModule = nullptr;
static PFN_cuInit                            g_cuInit = nullptr;
static PFN_cuDeviceGet                       g_cuDeviceGet = nullptr;
static PFN_cuDevicePrimaryCtxRetain          g_cuDevicePrimaryCtxRetain = nullptr;
static PFN_cuCtxSetCurrent                   g_cuCtxSetCurrent = nullptr;
static PFN_cuImportExternalMemory            g_cuImportExternalMemory = nullptr;
static PFN_cuExternalMemoryGetMappedBuffer   g_cuExternalMemoryGetMappedBuffer = nullptr;
static PFN_cuDestroyExternalMemory           g_cuDestroyExternalMemory = nullptr;
static PFN_cuMemcpyDtoH_v2                   g_cuMemcpyDtoH = nullptr;
static PFN_cuMemcpyDtoD_v2                   g_cuMemcpyDtoD = nullptr;
static PFN_cuMemcpyHtoD_v2                   g_cuMemcpyHtoD = nullptr;
static PFN_cuMemAlloc_v2                     g_cuMemAlloc = nullptr;
static PFN_cuMemFree_v2                      g_cuMemFree = nullptr;
static PFN_cuMemHostAlloc_v2                 g_cuMemHostAlloc = nullptr;
static PFN_cuMemFreeHost                     g_cuMemFreeHost = nullptr;
static PFN_cuDeviceGetLuid                   g_cuDeviceGetLuid = nullptr;
static PFN_cuCtxSynchronize                  g_cuCtxSynchronize = nullptr;
static PFN_cuMemcpyDtoDAsync_v2              g_cuMemcpyDtoDAsync = nullptr;
static PFN_cuMemcpyHtoDAsync_v2              g_cuMemcpyHtoDAsync = nullptr;
static PFN_cuStreamCreate                    g_cuStreamCreate = nullptr;
static PFN_cuStreamDestroy                   g_cuStreamDestroy = nullptr;
static PFN_cuEventCreate                     g_cuEventCreate = nullptr;
static PFN_cuEventRecord                     g_cuEventRecord = nullptr;
static PFN_cuEventSynchronize                g_cuEventSynchronize = nullptr;
static PFN_cuEventDestroy                    g_cuEventDestroy = nullptr;

static bool g_cudaInitialized = false;
static bool g_cudaInitAttempted = false;
static CUcontext g_cudaContext = nullptr;
static CUdevice g_cudaDevice = 0;

static bool EnsureDStorageLoaded() {
    if (g_DStorageGetFactory && g_DStorageCreateCompressionCodec) return true;

    // Get path to this DLL (dstorage_loader.dll)
    HMODULE thisModule = NULL;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCWSTR)&EnsureDStorageLoaded,
        &thisModule);

    WCHAR dllDir[MAX_PATH];
    GetModuleFileNameW(thisModule, dllDir, MAX_PATH);

    WCHAR* lastSlash = wcsrchr(dllDir, L'\\');
    if (lastSlash) *(lastSlash + 1) = 0;

    // Add our directory to DLL search path
    SetDllDirectoryW(dllDir);

    // CRITICAL: Pre-load dstoragecore.dll BEFORE dstorage.dll
    // When called from a DLL context, dstorage.dll's internal LoadLibrary
    // for dstoragecore.dll fails due to DLL search order differences.
    WCHAR corePath[MAX_PATH];
    wcscpy_s(corePath, dllDir);
    wcscat_s(corePath, L"dstoragecore.dll");
    g_dstorageCoreModule = LoadLibraryW(corePath);

    // Load dstorage.dll
    WCHAR dstoragePath[MAX_PATH];
    wcscpy_s(dstoragePath, dllDir);
    wcscat_s(dstoragePath, L"dstorage.dll");

    g_dstorageModule = LoadLibraryW(dstoragePath);
    if (!g_dstorageModule) {
        g_dstorageModule = LoadLibraryW(L"dstorage.dll");
    }
    if (!g_dstorageModule) return false;

    g_DStorageGetFactory = (PFN_DStorageGetFactory)GetProcAddress(g_dstorageModule, "DStorageGetFactory");
    g_DStorageCreateCompressionCodec = (PFN_DStorageCreateCompressionCodec)GetProcAddress(g_dstorageModule, "DStorageCreateCompressionCodec");
    return g_DStorageGetFactory != nullptr && g_DStorageCreateCompressionCodec != nullptr;
}

// ============================================================
// CUDA initialization (dynamic loading of nvcuda.dll)
// ============================================================

static bool EnsureCudaLoaded() {
    if (g_cudaInitialized) return true;
    if (g_cudaInitAttempted) return false;  // already tried, failed
    g_cudaInitAttempted = true;

    // nvcuda.dll is in System32 on any system with NVIDIA drivers
    g_nvcudaModule = LoadLibraryW(L"nvcuda.dll");
    if (!g_nvcudaModule) return false;

    // Load all function pointers
    g_cuInit = (PFN_cuInit)GetProcAddress(g_nvcudaModule, "cuInit");
    g_cuDeviceGet = (PFN_cuDeviceGet)GetProcAddress(g_nvcudaModule, "cuDeviceGet");
    g_cuDevicePrimaryCtxRetain = (PFN_cuDevicePrimaryCtxRetain)GetProcAddress(g_nvcudaModule, "cuDevicePrimaryCtxRetain");
    g_cuCtxSetCurrent = (PFN_cuCtxSetCurrent)GetProcAddress(g_nvcudaModule, "cuCtxSetCurrent");
    g_cuImportExternalMemory = (PFN_cuImportExternalMemory)GetProcAddress(g_nvcudaModule, "cuImportExternalMemory");
    g_cuExternalMemoryGetMappedBuffer = (PFN_cuExternalMemoryGetMappedBuffer)GetProcAddress(g_nvcudaModule, "cuExternalMemoryGetMappedBuffer");
    g_cuDestroyExternalMemory = (PFN_cuDestroyExternalMemory)GetProcAddress(g_nvcudaModule, "cuDestroyExternalMemory");
    g_cuMemcpyDtoH = (PFN_cuMemcpyDtoH_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyDtoH_v2");
    g_cuMemcpyDtoD = (PFN_cuMemcpyDtoD_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyDtoD_v2");
    g_cuMemcpyHtoD = (PFN_cuMemcpyHtoD_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyHtoD_v2");
    g_cuMemAlloc = (PFN_cuMemAlloc_v2)GetProcAddress(g_nvcudaModule, "cuMemAlloc_v2");
    g_cuMemFree = (PFN_cuMemFree_v2)GetProcAddress(g_nvcudaModule, "cuMemFree_v2");
    g_cuMemHostAlloc = (PFN_cuMemHostAlloc_v2)GetProcAddress(g_nvcudaModule, "cuMemHostAlloc");
    g_cuMemFreeHost = (PFN_cuMemFreeHost)GetProcAddress(g_nvcudaModule, "cuMemFreeHost");
    g_cuDeviceGetLuid = (PFN_cuDeviceGetLuid)GetProcAddress(g_nvcudaModule, "cuDeviceGetLuid");
    g_cuCtxSynchronize = (PFN_cuCtxSynchronize)GetProcAddress(g_nvcudaModule, "cuCtxSynchronize");

    g_cuMemcpyDtoDAsync = (PFN_cuMemcpyDtoDAsync_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyDtoDAsync_v2");
    if (!g_cuMemcpyDtoDAsync) {
        g_cuMemcpyDtoDAsync = (PFN_cuMemcpyDtoDAsync_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyDtoDAsync");
    }
    g_cuMemcpyHtoDAsync = (PFN_cuMemcpyHtoDAsync_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyHtoDAsync_v2");
    if (!g_cuMemcpyHtoDAsync) {
        g_cuMemcpyHtoDAsync = (PFN_cuMemcpyHtoDAsync_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyHtoDAsync");
    }
    g_cuStreamCreate = (PFN_cuStreamCreate)GetProcAddress(g_nvcudaModule, "cuStreamCreate");
    g_cuStreamDestroy = (PFN_cuStreamDestroy)GetProcAddress(g_nvcudaModule, "cuStreamDestroy");
    g_cuEventCreate = (PFN_cuEventCreate)GetProcAddress(g_nvcudaModule, "cuEventCreate");
    g_cuEventRecord = (PFN_cuEventRecord)GetProcAddress(g_nvcudaModule, "cuEventRecord");
    g_cuEventSynchronize = (PFN_cuEventSynchronize)GetProcAddress(g_nvcudaModule, "cuEventSynchronize");
    g_cuEventDestroy = (PFN_cuEventDestroy)GetProcAddress(g_nvcudaModule, "cuEventDestroy");

    if (!g_cuInit || !g_cuDeviceGet || !g_cuDevicePrimaryCtxRetain ||
        !g_cuCtxSetCurrent || !g_cuImportExternalMemory ||
        !g_cuExternalMemoryGetMappedBuffer || !g_cuDestroyExternalMemory ||
        !g_cuMemcpyDtoH || !g_cuMemcpyDtoD || !g_cuMemcpyHtoD ||
        !g_cuMemAlloc || !g_cuMemFree || !g_cuMemHostAlloc ||
        !g_cuMemFreeHost || !g_cuDeviceGetLuid || !g_cuCtxSynchronize) {
        return false;
    }

    // Initialize CUDA and get a context on device 0
    CUresult cr = g_cuInit(0);
    if (cr != CUDA_SUCCESS) return false;

    cr = g_cuDeviceGet(&g_cudaDevice, 0);
    if (cr != CUDA_SUCCESS) return false;

    cr = g_cuDevicePrimaryCtxRetain(&g_cudaContext, g_cudaDevice);
    if (cr != CUDA_SUCCESS) return false;

    cr = g_cuCtxSetCurrent(g_cudaContext);
    if (cr != CUDA_SUCCESS) return false;

    g_cudaInitialized = true;
    return true;
}

// Helper: submit queue, wait for fence, check errors
static HRESULT SubmitAndWait(IDStorageQueue* queue, ID3D12Device* device) {
    ComPtr<ID3D12Fence> fence;
    HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) return hr;

    HANDLE event = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!event) return HRESULT_FROM_WIN32(GetLastError());

    hr = fence->SetEventOnCompletion(1, event);
    if (FAILED(hr)) { CloseHandle(event); return hr; }

    queue->EnqueueSignal(fence.Get(), 1);
    queue->Submit();

    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);

    DSTORAGE_ERROR_RECORD errorRecord = {};
    queue->RetrieveErrorRecord(&errorRecord);
    if (FAILED(errorRecord.FirstFailure.HResult)) {
        return errorRecord.FirstFailure.HResult;
    }
    return S_OK;
}

struct DSLoader {
    ComPtr<IDStorageFactory> factory;
    ComPtr<IDStorageQueue> queue;
    ComPtr<IDStorageQueue> memQueue;  // Separate queue for memory destination reads
    ComPtr<ID3D12Device> device;

    // Cached file handle for batched operations
    ComPtr<IDStorageFile> cachedFile;
    std::wstring cachedFilePath;

    // Persistent fence for async submit/wait (prefetching)
    ComPtr<ID3D12Fence> asyncFence;
    HANDLE asyncEvent;
    UINT64 asyncFenceValue;   // increments on each submit
    bool asyncPending;         // true if a submit is in-flight

    // Track shared heaps for D3D12-CUDA interop:
    // Maps ID3D12Resource* -> (heap, heapSize) for shared placed resources.
    // The heap must live as long as the resource.
    struct SharedHeapEntry {
        ComPtr<ID3D12Heap> heap;
        uint64_t heapSize;
    };
    std::unordered_map<ID3D12Resource*, SharedHeapEntry> sharedHeaps;

    // Reusable staging buffer for stream-to-cuda operations.
    // Grows as needed, never shrinks. Destroyed with the loader.
    struct {
        ComPtr<ID3D12Heap> heap;
        ComPtr<ID3D12Resource> resource;
        CUexternalMemory extMem = nullptr;
        CUdeviceptr devPtr = 0;
        HANDLE sharedHandle = nullptr;
        uint64_t heapSize = 0;    // actual heap size (64KB aligned)
        uint64_t dataSize = 0;    // usable data capacity
    } staging;

    void* cudaStream = nullptr;
    void* copyEvent = nullptr;
};

struct CUDAInterop {
    CUexternalMemory extMem;   // CUDA external memory handle
    CUdeviceptr devPtr;        // CUDA device pointer (accessible by CUDA/GGML)
    HANDLE sharedHandle;       // NT handle from ID3D12Device::CreateSharedHandle
    uint64_t size;             // buffer size in bytes
};

static HRESULT g_lastHR = S_OK;

static bool DStorageLoaderDebugEnabled() {
    const char* value = std::getenv("LLAMA_DSTORAGE_DEBUG");
    return value != nullptr &&
           value[0] != '\0' &&
           std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

static uint64_t DStorageLoaderTimeUS() {
    LARGE_INTEGER freq = {};
    LARGE_INTEGER counter = {};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000ULL) / freq.QuadPart);
}

static IDStorageFile* GetCachedFile(DSLoader* loader, const wchar_t* file_path) {
    if (!loader || !file_path) {
        return nullptr;
    }

    if (loader->cachedFile && loader->cachedFilePath == file_path) {
        return loader->cachedFile.Get();
    }

    loader->cachedFile.Reset();
    loader->cachedFilePath = file_path;

    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&loader->cachedFile));
    if (FAILED(hr)) {
        g_lastHR = hr;
        loader->cachedFilePath.clear();
        return nullptr;
    }

    return loader->cachedFile.Get();
}

extern "C" {

int32_t ds_loader_get_hresult() {
    return (int32_t)g_lastHR;
}

int ds_loader_available() {
    CoInitializeEx(NULL, COINIT_MULTITHREADED);

    if (!EnsureDStorageLoaded()) {
        printf("DEBUG: EnsureDStorageLoaded failed\n");
        g_lastHR = E_FAIL;
        return 0;
    }

    ComPtr<ID3D12Device> device;
    g_lastHR = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device));
    if (FAILED(g_lastHR)) {
        printf("DEBUG: D3D12CreateDevice failed with hr=0x%08x\n", (uint32_t)g_lastHR);
        return 0;
    }

    IDStorageFactory* rawFactory = nullptr;
    g_lastHR = g_DStorageGetFactory(__uuidof(IDStorageFactory), (void**)&rawFactory);
    if (rawFactory) rawFactory->Release();
    if (FAILED(g_lastHR)) {
        printf("DEBUG: g_DStorageGetFactory failed with hr=0x%08x\n", (uint32_t)g_lastHR);
    }
    return SUCCEEDED(g_lastHR) ? 1 : 0;
}

DSLoaderHandle ds_loader_create() {
    CoInitializeEx(NULL, COINIT_MULTITHREADED);

    if (!EnsureDStorageLoaded()) {
        printf("DEBUG: ds_loader_create: EnsureDStorageLoaded failed\n");
        return NULL;
    }

    DSLoader* loader = new (std::nothrow) DSLoader();
    if (!loader) return NULL;

    // LUID matching: if CUDA is available, find the DXGI adapter that matches
    // the CUDA device's LUID. This ensures D3D12 and CUDA use the same GPU,
    // which is required for D3D12-CUDA interop (cuImportExternalMemory).
    // On laptops with Intel iGPU + NVIDIA dGPU, D3D12CreateDevice(NULL, ...)
    // may pick the Intel iGPU while CUDA device 0 is the NVIDIA GPU.
    HRESULT hr = E_FAIL;
    bool deviceCreated = false;

    if (EnsureCudaLoaded() && g_cuDeviceGetLuid) {
        // Get CUDA device's LUID
        char cudaLuid[8] = {};
        unsigned int cudaNodeMask = 0;
        CUresult cr = g_cuDeviceGetLuid(cudaLuid, &cudaNodeMask, g_cudaDevice);

        if (cr == CUDA_SUCCESS) {
            // Create DXGI factory to enumerate adapters
            typedef HRESULT (WINAPI *PFN_CreateDXGIFactory2)(UINT, REFIID, void**);
            HMODULE dxgiMod = GetModuleHandleW(L"dxgi.dll");
            if (!dxgiMod) dxgiMod = LoadLibraryW(L"dxgi.dll");

            if (dxgiMod) {
                PFN_CreateDXGIFactory2 pfnCreateFactory =
                    (PFN_CreateDXGIFactory2)GetProcAddress(dxgiMod, "CreateDXGIFactory2");

                if (pfnCreateFactory) {
                    ComPtr<IDXGIFactory4> dxgiFactory;
                    hr = pfnCreateFactory(0, IID_PPV_ARGS(&dxgiFactory));

                    if (SUCCEEDED(hr)) {
                        // Convert CUDA LUID bytes to LUID struct
                        LUID adapterLuid;
                        memcpy(&adapterLuid, cudaLuid, sizeof(LUID));

                        // Find the adapter matching CUDA's LUID
                        ComPtr<IDXGIAdapter> adapter;
                        hr = dxgiFactory->EnumAdapterByLuid(adapterLuid, IID_PPV_ARGS(&adapter));

                        if (SUCCEEDED(hr)) {
                            // Create D3D12 device on the CUDA-matching adapter
                            hr = D3D12CreateDevice(
                                adapter.Get(),
                                D3D_FEATURE_LEVEL_12_1,
                                IID_PPV_ARGS(&loader->device));

                            if (SUCCEEDED(hr)) {
                                deviceCreated = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: if LUID matching failed (no CUDA, or adapter not found),
    // use the default adapter. D3D12-CUDA interop may not work, but
    // DirectStorage SSD→GPU streaming will still function.
    if (!deviceCreated) {
        hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&loader->device));
    }

    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    hr = g_DStorageGetFactory(__uuidof(IDStorageFactory), (void**)&loader->factory);
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    DSTORAGE_QUEUE_DESC queueDesc = {};
    queueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device = loader->device.Get();

    hr = loader->factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&loader->queue));
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    DSTORAGE_QUEUE_DESC memQueueDesc = {};
    memQueueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    memQueueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    memQueueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    memQueueDesc.Device = loader->device.Get();

    hr = loader->factory->CreateQueue(&memQueueDesc, IID_PPV_ARGS(&loader->memQueue));
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    hr = loader->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&loader->asyncFence));
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }
    loader->asyncEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!loader->asyncEvent) {
        g_lastHR = HRESULT_FROM_WIN32(GetLastError());
        delete loader;
        return NULL;
    }
    loader->asyncFenceValue = 0;
    loader->asyncPending = false;

    if (EnsureCudaLoaded()) {
        g_cuCtxSetCurrent(g_cudaContext);
        if (g_cuStreamCreate) {
            g_cuStreamCreate(&loader->cudaStream, 1); // 1 = CU_STREAM_NON_BLOCKING
        }
        if (g_cuEventCreate) {
            g_cuEventCreate(&loader->copyEvent, 2); // 2 = CU_EVENT_DISABLE_TIMING
        }
    }

    return loader;
}

static void DestroyStagingBuffer(DSLoader* loader) {
    if (!loader) return;
    if (g_cudaInitialized) {
        g_cuCtxSetCurrent(g_cudaContext);
        if (loader->staging.devPtr) {
            g_cuMemFree(loader->staging.devPtr);
            loader->staging.devPtr = 0;
        }
        if (loader->staging.extMem) {
            g_cuDestroyExternalMemory(loader->staging.extMem);
            loader->staging.extMem = nullptr;
        }
    }
    if (loader->staging.sharedHandle) {
        CloseHandle(loader->staging.sharedHandle);
        loader->staging.sharedHandle = nullptr;
    }
    loader->staging.resource.Reset();
    loader->staging.heap.Reset();
    loader->staging.heapSize = 0;
    loader->staging.dataSize = 0;
}

static bool EnsureStagingBuffer(DSLoader* loader, uint64_t minSize) {
    if (!loader || minSize == 0) return false;
    if (loader->staging.dataSize >= minSize) return true;  // already big enough

    if (!EnsureCudaLoaded()) return false;

    // Destroy old staging buffer if it exists
    DestroyStagingBuffer(loader);

    // Align heap size to 64KB
    uint64_t heapSize = (minSize + 65535) & ~65535ULL;

    // Create shared D3D12 heap
    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = heapSize;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Properties.CreationNodeMask = 1;
    hd.Properties.VisibleNodeMask = 1;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = (D3D12_HEAP_FLAGS)(D3D12_HEAP_FLAG_SHARED |
               D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES |
               D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES);

    HRESULT hr = loader->device->CreateHeap(&hd, IID_PPV_ARGS(&loader->staging.heap));
    if (FAILED(hr)) { g_lastHR = hr; return false; }

    // Create placed resource on the shared heap
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = minSize;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = loader->device->CreatePlacedResource(
        loader->staging.heap.Get(), 0, &rd,
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&loader->staging.resource));
    if (FAILED(hr)) { g_lastHR = hr; DestroyStagingBuffer(loader); return false; }

    // Create shared NT handle from the heap
    hr = loader->device->CreateSharedHandle(
        loader->staging.heap.Get(), nullptr, GENERIC_ALL, nullptr,
        &loader->staging.sharedHandle);
    if (FAILED(hr)) { g_lastHR = hr; DestroyStagingBuffer(loader); return false; }

    // Import into CUDA
    g_cuCtxSetCurrent(g_cudaContext);

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 memDesc = {};
    memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
    memDesc.handle.win32.handle = loader->staging.sharedHandle;
    memDesc.size = heapSize;
    memDesc.flags = 0;

    CUresult cr = g_cuImportExternalMemory(&loader->staging.extMem, &memDesc);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCA000000 | (cr & 0xFFFF));
        DestroyStagingBuffer(loader);
        return false;
    }

    // Map to CUDA device pointer
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = minSize;
    bufDesc.flags = 0;

    cr = g_cuExternalMemoryGetMappedBuffer(&loader->staging.devPtr, loader->staging.extMem, &bufDesc);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCB000000 | (cr & 0xFFFF));
        DestroyStagingBuffer(loader);
        return false;
    }

    loader->staging.heapSize = heapSize;
    loader->staging.dataSize = minSize;
    return true;
}

void ds_loader_destroy(DSLoaderHandle loader) {
    if (loader) {
        DestroyStagingBuffer(loader);
        if (g_cudaInitialized) {
            g_cuCtxSetCurrent(g_cudaContext);
            if (loader->cudaStream && g_cuStreamDestroy) {
                g_cuStreamDestroy(loader->cudaStream);
                loader->cudaStream = nullptr;
            }
            if (loader->copyEvent && g_cuEventDestroy) {
                g_cuEventDestroy(loader->copyEvent);
                loader->copyEvent = nullptr;
            }
        }
        if (loader->asyncEvent) CloseHandle(loader->asyncEvent);
        delete loader;
    }
}

// --- GPU buffer management ---

void* ds_loader_create_gpu_buffer(DSLoaderHandle loader, uint64_t size) {
    if (!loader || size == 0) return NULL;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.SampleDesc.Count = 1;

    ID3D12Resource* resource = nullptr;
    HRESULT hr = loader->device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&resource));

    if (FAILED(hr)) {
        g_lastHR = hr;
        return NULL;
    }
    return resource;
}

void ds_loader_destroy_gpu_buffer(void* gpu_buffer) {
    if (gpu_buffer) {
        ((ID3D12Resource*)gpu_buffer)->Release();
    }
}

// --- File reads ---

int ds_loader_read(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer
) {
    if (!loader || !file_path || !gpu_buffer || size == 0) return -1;

    ComPtr<IDStorageFile> file;
    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&file));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    request.Source.File.Source = file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = (uint32_t)size;
    request.UncompressedSize = (uint32_t)size;
    request.Destination.Buffer.Resource = (ID3D12Resource*)gpu_buffer;
    request.Destination.Buffer.Offset = 0;
    request.Destination.Buffer.Size = (uint32_t)size;

    loader->queue->EnqueueRequest(&request);

    hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    if (FAILED(hr)) { g_lastHR = hr; return -1; }
    return 0;
}

int ds_loader_read_to_memory(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* dest_memory
) {
    if (!loader || !file_path || !dest_memory || size == 0) return -1;

    ComPtr<IDStorageFile> file;
    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&file));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
    request.Source.File.Source = file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = (uint32_t)size;
    request.UncompressedSize = (uint32_t)size;
    request.Destination.Memory.Buffer = dest_memory;
    request.Destination.Memory.Size = (uint32_t)size;

    loader->memQueue->EnqueueRequest(&request);

    hr = SubmitAndWait(loader->memQueue.Get(), loader->device.Get());
    if (FAILED(hr)) { g_lastHR = hr; return -1; }
    return 0;
}

// --- Chunked read (for tensors > 32MB) ---

#define DS_MAX_CHUNK_SIZE (32ULL * 1024 * 1024)  // 32 MB per request

int ds_loader_read_chunked(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t total_size,
    void* gpu_buffer
) {
    if (!loader || !file_path || !gpu_buffer || total_size == 0) return -1;

    ComPtr<IDStorageFile> file;
    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&file));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    // Enqueue all chunks before submitting
    uint64_t remaining = total_size;
    uint64_t bufferOffset = 0;

    while (remaining > 0) {
        uint32_t chunkSize = (remaining > DS_MAX_CHUNK_SIZE)
            ? (uint32_t)DS_MAX_CHUNK_SIZE
            : (uint32_t)remaining;

        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        request.Source.File.Source = file.Get();
        request.Source.File.Offset = file_offset + bufferOffset;
        request.Source.File.Size = chunkSize;
        request.UncompressedSize = chunkSize;
        request.Destination.Buffer.Resource = (ID3D12Resource*)gpu_buffer;
        request.Destination.Buffer.Offset = bufferOffset;
        request.Destination.Buffer.Size = chunkSize;

        loader->queue->EnqueueRequest(&request);

        bufferOffset += chunkSize;
        remaining -= chunkSize;
    }

    // Single submit + wait for all chunks
    hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    if (FAILED(hr)) { g_lastHR = hr; return -1; }
    return 0;
}

// --- Batched reads (file handle caching + multi-request submit) ---

int ds_loader_open_file(DSLoaderHandle loader, const wchar_t* file_path) {
    if (!loader || !file_path) return -1;

    // If same file is already cached, reuse it
    if (loader->cachedFile && loader->cachedFilePath == file_path) {
        return 0;
    }

    // Release any previously cached file
    loader->cachedFile.Reset();
    loader->cachedFilePath.clear();

    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&loader->cachedFile));
    if (FAILED(hr)) {
        g_lastHR = hr;
        return -1;
    }

    loader->cachedFilePath = file_path;
    return 0;
}

void ds_loader_close_file(DSLoaderHandle loader) {
    if (!loader) return;
    loader->cachedFile.Reset();
    loader->cachedFilePath.clear();
}

int ds_loader_enqueue_read(
    DSLoaderHandle loader,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer,
    uint64_t buffer_offset
) {
    if (!loader || !gpu_buffer || size == 0) return -1;
    if (!loader->cachedFile) {
        g_lastHR = E_HANDLE;
        return -1;
    }

    // Auto-chunk if > 32MB
    uint64_t remaining = size;
    uint64_t srcOffset = file_offset;
    uint64_t dstOffset = buffer_offset;

    while (remaining > 0) {
        uint32_t chunkSize = (remaining > DS_MAX_CHUNK_SIZE)
            ? (uint32_t)DS_MAX_CHUNK_SIZE
            : (uint32_t)remaining;

        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        request.Source.File.Source = loader->cachedFile.Get();
        request.Source.File.Offset = srcOffset;
        request.Source.File.Size = chunkSize;
        request.UncompressedSize = chunkSize;
        request.Destination.Buffer.Resource = (ID3D12Resource*)gpu_buffer;
        request.Destination.Buffer.Offset = dstOffset;
        request.Destination.Buffer.Size = chunkSize;

        loader->queue->EnqueueRequest(&request);

        srcOffset += chunkSize;
        dstOffset += chunkSize;
        remaining -= chunkSize;
    }

    return 0;
}

int ds_loader_submit_and_wait(DSLoaderHandle loader) {
    if (!loader) return -1;

    HRESULT hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    if (FAILED(hr)) {
        g_lastHR = hr;
        return -1;
    }
    return 0;
}

// --- Async submit for prefetching ---

int ds_loader_submit(DSLoaderHandle loader) {
    if (!loader) return -1;

    // If a previous async submit is still pending, wait for it first
    if (loader->asyncPending) {
        WaitForSingleObject(loader->asyncEvent, INFINITE);
        DSTORAGE_ERROR_RECORD errorRecord = {};
        loader->queue->RetrieveErrorRecord(&errorRecord);
        if (FAILED(errorRecord.FirstFailure.HResult)) {
            g_lastHR = errorRecord.FirstFailure.HResult;
            loader->asyncPending = false;
            return -1;
        }
        loader->asyncPending = false;
    }

    // Increment fence value, enqueue signal, submit — return immediately
    loader->asyncFenceValue++;
    HRESULT hr = loader->asyncFence->SetEventOnCompletion(loader->asyncFenceValue, loader->asyncEvent);
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    loader->queue->EnqueueSignal(loader->asyncFence.Get(), loader->asyncFenceValue);
    loader->queue->Submit();
    loader->asyncPending = true;

    return 0;
}

int ds_loader_is_complete(DSLoaderHandle loader) {
    if (!loader) return 1;
    if (!loader->asyncPending) return 1;

    UINT64 completed = loader->asyncFence->GetCompletedValue();
    return (completed >= loader->asyncFenceValue) ? 1 : 0;
}

int ds_loader_wait_complete(DSLoaderHandle loader) {
    if (!loader) return -1;
    if (!loader->asyncPending) return 0;

    WaitForSingleObject(loader->asyncEvent, INFINITE);
    loader->asyncPending = false;

    // Check for errors
    DSTORAGE_ERROR_RECORD errorRecord = {};
    loader->queue->RetrieveErrorRecord(&errorRecord);
    if (FAILED(errorRecord.FirstFailure.HResult)) {
        g_lastHR = errorRecord.FirstFailure.HResult;
        return -1;
    }
    return 0;
}

// --- GPU readback ---

int ds_loader_gpu_readback(
    DSLoaderHandle loader,
    void* gpu_buffer,
    uint64_t size,
    void* dest_memory
) {
    if (!loader || !gpu_buffer || !dest_memory || size == 0) return -1;

    ID3D12Resource* srcResource = (ID3D12Resource*)gpu_buffer;

    // Create readback buffer (CPU-readable)
    D3D12_HEAP_PROPERTIES readbackHeap = {};
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.SampleDesc.Count = 1;

    ComPtr<ID3D12Resource> readbackResource;
    HRESULT hr = loader->device->CreateCommittedResource(
        &readbackHeap,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackResource));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    // Create command queue + command list for the copy
    D3D12_COMMAND_QUEUE_DESC cmdQueueDesc = {};
    cmdQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;

    ComPtr<ID3D12CommandQueue> cmdQueue;
    hr = loader->device->CreateCommandQueue(&cmdQueueDesc, IID_PPV_ARGS(&cmdQueue));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    ComPtr<ID3D12CommandAllocator> cmdAlloc;
    hr = loader->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&cmdAlloc));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    ComPtr<ID3D12GraphicsCommandList> cmdList;
    hr = loader->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, cmdAlloc.Get(), nullptr, IID_PPV_ARGS(&cmdList));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    // Record copy command
    cmdList->CopyBufferRegion(readbackResource.Get(), 0, srcResource, 0, size);
    cmdList->Close();

    // Execute
    ID3D12CommandList* lists[] = { cmdList.Get() };
    cmdQueue->ExecuteCommandLists(1, lists);

    // Fence + wait
    ComPtr<ID3D12Fence> fence;
    hr = loader->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    HANDLE event = CreateEvent(NULL, FALSE, FALSE, NULL);
    fence->SetEventOnCompletion(1, event);
    cmdQueue->Signal(fence.Get(), 1);
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);

    // Map readback buffer and copy to dest
    void* mapped = nullptr;
    D3D12_RANGE readRange = { 0, (SIZE_T)size };
    hr = readbackResource->Map(0, &readRange, &mapped);
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    memcpy(dest_memory, mapped, (size_t)size);

    D3D12_RANGE writeRange = { 0, 0 };
    readbackResource->Unmap(0, &writeRange);

    return 0;
}

// --- Diagnostic for shared heap support ---

// Returns bitmask: bits 0-7 = ResourceHeapTier, bit 8 = heap(SHARED), bit 9 = heap(SHARED+DENY),
// bit 10 = placed(no flags), bit 11 = placed(SIMULTANEOUS_ACCESS), bit 12 = committed(SHARED).
// g_lastHR set to first failed HRESULT.
int ds_loader_debug_shared(DSLoaderHandle loader) {
    if (!loader) return -1;

    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    loader->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));

    int result = (int)(options.ResourceHeapTier & 0xFF);

    // Test 1: CreateHeap with SHARED only
    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = 65536;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Properties.CreationNodeMask = 1;
    hd.Properties.VisibleNodeMask = 1;
    hd.Alignment = 65536;
    hd.Flags = D3D12_HEAP_FLAG_SHARED;

    ComPtr<ID3D12Heap> h1;
    HRESULT hr = loader->device->CreateHeap(&hd, IID_PPV_ARGS(&h1));
    if (SUCCEEDED(hr)) result |= (1 << 8);
    else g_lastHR = hr;

    // Test 2: CreateHeap with SHARED + buffer-only deny flags
    hd.Flags = (D3D12_HEAP_FLAGS)(D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES);
    ComPtr<ID3D12Heap> h2;
    hr = loader->device->CreateHeap(&hd, IID_PPV_ARGS(&h2));
    if (SUCCEEDED(hr)) result |= (1 << 9);

    // Test 3 & 4: CreatePlacedResource on whichever heap worked
    ID3D12Heap* heap = h1 ? h1.Get() : (h2 ? h2.Get() : nullptr);
    if (heap) {
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 65536;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.SampleDesc.Count = 1;

        rd.Flags = D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> r3;
        hr = loader->device->CreatePlacedResource(heap, 0, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r3));
        if (SUCCEEDED(hr)) result |= (1 << 10);
        else if (!(result & 0xFF00)) g_lastHR = hr;

        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        ComPtr<ID3D12Resource> r4;
        hr = loader->device->CreatePlacedResource(heap, 0, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r4));
        if (SUCCEEDED(hr)) result |= (1 << 11);
    }

    // Test 5: CreateCommittedResource with SHARED
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    hp.CreationNodeMask = 1;
    hp.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC rd2 = {};
    rd2.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd2.Width = 65536;
    rd2.Height = 1;
    rd2.DepthOrArraySize = 1;
    rd2.MipLevels = 1;
    rd2.Format = DXGI_FORMAT_UNKNOWN;
    rd2.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd2.SampleDesc.Count = 1;
    rd2.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    ComPtr<ID3D12Resource> r5;
    hr = loader->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd2, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r5));
    if (SUCCEEDED(hr)) result |= (1 << 12);

    return result;
}

// --- CUDA interop (D3D12 <-> CUDA shared memory) ---

int ds_loader_cuda_available() {
    return EnsureCudaLoaded() ? 1 : 0;
}

void* ds_loader_create_shared_gpu_buffer(DSLoaderHandle loader, uint64_t size) {
    if (!loader || size == 0) return NULL;

    // Align heap size up to 64KB (D3D12 resource placement alignment)
    uint64_t heapSize = (size + 65535) & ~65535ULL;

    // Step 1: Create a shared D3D12 heap
    D3D12_HEAP_DESC heapDesc = {};
    heapDesc.SizeInBytes = heapSize;
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapDesc.Properties.CreationNodeMask = 1;
    heapDesc.Properties.VisibleNodeMask = 1;
    heapDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT; // 64KB
    heapDesc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;

    ComPtr<ID3D12Heap> heap;
    HRESULT hr = loader->device->CreateHeap(&heapDesc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        g_lastHR = hr;
        return NULL;
    }

    // Step 2: Create a placed resource on the shared heap
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* resource = nullptr;
    hr = loader->device->CreatePlacedResource(
        heap.Get(), 0, &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&resource));

    if (FAILED(hr)) {
        g_lastHR = hr;
        return NULL;
    }

    // Track the heap alongside the resource for CUDA export
    loader->sharedHeaps[resource] = { heap, heapSize };

    return resource;
}

CUDAInteropHandle ds_loader_export_to_cuda(DSLoaderHandle loader, void* shared_gpu_buffer, uint64_t size) {
    if (!loader || !shared_gpu_buffer || size == 0) return NULL;
    if (!EnsureCudaLoaded()) return NULL;

    ID3D12Resource* resource = (ID3D12Resource*)shared_gpu_buffer;

    // Look up the shared heap for this resource
    auto it = loader->sharedHeaps.find(resource);
    if (it == loader->sharedHeaps.end()) {
        // Not a shared buffer — must use ds_loader_create_shared_gpu_buffer
        g_lastHR = E_HANDLE;
        return NULL;
    }
    ID3D12Heap* heap = it->second.heap.Get();
    uint64_t heapSize = it->second.heapSize;

    // Step 1: Create shared NT handle from the D3D12 HEAP
    HANDLE sharedHandle = NULL;
    HRESULT hr = loader->device->CreateSharedHandle(
        heap,
        nullptr,        // security attributes
        GENERIC_ALL,    // access
        nullptr,        // name
        &sharedHandle);

    if (FAILED(hr) || !sharedHandle) {
        g_lastHR = hr ? hr : E_HANDLE;
        return NULL;
    }

    // Step 2: Ensure CUDA context is current
    CUresult cr = g_cuCtxSetCurrent(g_cudaContext);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCC000000 | (cr & 0xFFFF));  // CC = CUDA context error
        CloseHandle(sharedHandle);
        return NULL;
    }

    // Step 3: Import D3D12 HEAP into CUDA as external memory
    //         (D3D12_HEAP type, NOT D3D12_RESOURCE — heap-based interop)
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 memDesc = {};
    memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
    memDesc.handle.win32.handle = sharedHandle;
    memDesc.handle.win32.name = nullptr;
    memDesc.size = heapSize;  // must match the actual heap size (64KB-aligned)
    memDesc.flags = 0;        // NOT CUDA_EXTERNAL_MEMORY_DEDICATED for heaps

    CUexternalMemory extMem = nullptr;
    cr = g_cuImportExternalMemory(&extMem, &memDesc);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCA000000 | (cr & 0xFFFF));  // CA = CUDA import error
        CloseHandle(sharedHandle);
        return NULL;
    }

    // Step 4: Map a buffer from the imported heap memory to a CUDA device pointer
    //         The placed resource starts at offset 0 and has 'size' bytes of data
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = size;      // original data size, not the aligned heap size
    bufDesc.flags = 0;

    CUdeviceptr devPtr = 0;
    cr = g_cuExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCB000000 | (cr & 0xFFFF));  // CB = CUDA map error
        g_cuDestroyExternalMemory(extMem);
        CloseHandle(sharedHandle);
        return NULL;
    }

    // Step 5: Package into opaque interop handle
    CUDAInterop* interop = new (std::nothrow) CUDAInterop();
    if (!interop) {
        g_cuMemFree(devPtr);
        g_cuDestroyExternalMemory(extMem);
        CloseHandle(sharedHandle);
        return NULL;
    }

    interop->extMem = extMem;
    interop->devPtr = devPtr;
    interop->sharedHandle = sharedHandle;
    interop->size = size;

    return interop;
}

uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle interop) {
    if (!interop) return 0;
    return interop->devPtr;
}

int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle interop, void* dest, uint64_t size) {
    if (!interop || !dest || size == 0) return -1;
    if (!g_cudaInitialized) return -1;

    CUresult cr = g_cuCtxSetCurrent(g_cudaContext);
    if (cr != CUDA_SUCCESS) return -1;

    cr = g_cuMemcpyDtoH(dest, interop->devPtr, (size_t)size);
    if (cr != CUDA_SUCCESS) return -1;

    return 0;
}

// --- Stream file data directly to a CUDA device pointer ---
// Uses a reusable staging buffer (D3D12 shared heap + CUDA interop).
// Path: SSD -> DirectStorage DMA -> staging GPU buffer -> cuMemcpyDtoD -> dest CUDA ptr
// Zero CPU copies. Zero Go allocations.
int ds_loader_stream_to_cuda(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    uint64_t cuda_dest_ptr
) {
    if (!loader || !file_path || size == 0 || cuda_dest_ptr == 0) return -1;
    if (!EnsureCudaLoaded()) { g_lastHR = E_FAIL; return -1; }

    // DSTORAGE requires unbuffered file alignment (typically 4K) for the source offset
    // when destination is a memory buffer. GGUF offsets are typically only 32-byte aligned.
    const uint64_t align_offset = file_offset % 4096;
    const uint64_t aligned_file_offset = file_offset - align_offset;
    const uint64_t aligned_size = size + align_offset;

    // Step 1: Ensure staging buffer is large enough
    if (!EnsureStagingBuffer(loader, aligned_size)) return -1;

    // Step 2: Load file data into staging buffer via DirectStorage
    ComPtr<IDStorageFile> file;
    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&file));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    uint64_t remaining = aligned_size;
    uint64_t bufferOffset = 0;
    while (remaining > 0) {
        uint32_t chunkSize = (remaining > DS_MAX_CHUNK_SIZE)
            ? (uint32_t)DS_MAX_CHUNK_SIZE
            : (uint32_t)remaining;

        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        request.Source.File.Source = file.Get();
        request.Source.File.Offset = aligned_file_offset + bufferOffset;
        request.Source.File.Size = chunkSize;
        request.UncompressedSize = chunkSize;
        request.Destination.Buffer.Resource = loader->staging.resource.Get();
        request.Destination.Buffer.Offset = bufferOffset;
        request.Destination.Buffer.Size = chunkSize;

        loader->queue->EnqueueRequest(&request);
        bufferOffset += chunkSize;
        remaining -= chunkSize;
    }

    hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    // Step 3: Device-to-device copy from staging CUDA ptr to destination,
    // skipping the padding added to achieve 4K alignment.
    g_cuCtxSetCurrent(g_cudaContext);
    CUresult cr = g_cuMemcpyDtoD((CUdeviceptr)cuda_dest_ptr, loader->staging.devPtr + align_offset, (size_t)size);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCD000000 | (cr & 0xFFFF));
        return -1;
    }

    if (g_cuCtxSynchronize) {
        g_cuCtxSynchronize();
    }

    return 0;
}

int ds_loader_stream_to_cuda_batch(
    DSLoaderHandle loader,
    const DSLoaderStreamRequest* requests,
    int request_count
) {
    if (!loader || !requests || request_count <= 0) return -1;
    if (!EnsureCudaLoaded()) { g_lastHR = E_FAIL; return -1; }
    const bool debug = DStorageLoaderDebugEnabled();
    const uint64_t total_t0 = debug ? DStorageLoaderTimeUS() : 0;

    struct BatchSegment {
        uint64_t aligned_file_offset = 0;
        uint64_t align_offset = 0;        // non-zero only for uncompressed requests
        uint64_t aligned_size = 0;
        uint64_t staging_offset = 0;
        bool is_compressed = false;
    };

    std::vector<BatchSegment> segments;
    segments.reserve((size_t) request_count);

    uint64_t staging_size = 0;
    for (int i = 0; i < request_count; ++i) {
        const DSLoaderStreamRequest & req = requests[i];
        if (!req.file_path || req.size == 0 || req.cuda_dest_ptr == 0) {
            return -1;
        }

        BatchSegment seg = {};
        seg.is_compressed = req.uncompressed_size > req.size;

        if (seg.is_compressed) {
            // Compressed (GDeflate) requests: source offset is already 4K-aligned
            // by the compressor. No alignment padding needed. The decompressed
            // output goes directly at staging_offset with no skip.
            seg.align_offset = 0;
            seg.aligned_file_offset = req.file_offset;
            seg.aligned_size = req.uncompressed_size;
        } else {
            // Uncompressed requests: align source to 4K for DirectStorage BypassIO
            seg.align_offset = req.file_offset % 4096;
            seg.aligned_file_offset = req.file_offset - seg.align_offset;
            seg.aligned_size = req.size + seg.align_offset;
        }
        seg.staging_offset = (staging_size + 4095) & ~4095ULL;
        staging_size = seg.staging_offset + seg.aligned_size;
        segments.push_back(seg);
    }

    if (!EnsureStagingBuffer(loader, staging_size)) return -1;
    const uint64_t after_staging_us = debug ? DStorageLoaderTimeUS() : 0;

    for (int i = 0; i < request_count; ++i) {
        IDStorageFile* file = GetCachedFile(loader, requests[i].file_path);
        if (!file) { return -1; }

        const bool is_compressed = requests[i].uncompressed_size > requests[i].size;
        if (is_compressed) {
            DSTORAGE_REQUEST request = {};
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
            request.Source.File.Source = file;
            request.Source.File.Offset = segments[(size_t) i].aligned_file_offset;
            request.Source.File.Size = (uint32_t) ((requests[i].size + 4095) & ~4095ULL);
            request.UncompressedSize = (uint32_t) requests[i].uncompressed_size;
            request.Destination.Buffer.Resource = loader->staging.resource.Get();
            request.Destination.Buffer.Offset = segments[(size_t) i].staging_offset;
            request.Destination.Buffer.Size = (uint32_t) requests[i].uncompressed_size;

            loader->queue->EnqueueRequest(&request);
        } else {
            uint64_t remaining = segments[(size_t) i].aligned_size;
            uint64_t source_offset = 0;
            while (remaining > 0) {
                uint32_t chunk_size = (remaining > DS_MAX_CHUNK_SIZE)
                    ? (uint32_t) DS_MAX_CHUNK_SIZE
                    : (uint32_t) remaining;

                DSTORAGE_REQUEST request = {};
                request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                request.Source.File.Source = file;
                request.Source.File.Offset = segments[(size_t) i].aligned_file_offset + source_offset;
                request.Source.File.Size = chunk_size;
                request.UncompressedSize = chunk_size;
                request.Destination.Buffer.Resource = loader->staging.resource.Get();
                request.Destination.Buffer.Offset = segments[(size_t) i].staging_offset + source_offset;
                request.Destination.Buffer.Size = chunk_size;

                loader->queue->EnqueueRequest(&request);
                source_offset += chunk_size;
                remaining -= chunk_size;
            }
        }
    }

    const uint64_t submit_t0 = debug ? DStorageLoaderTimeUS() : 0;
    HRESULT hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    const uint64_t submit_t1 = debug ? DStorageLoaderTimeUS() : 0;
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    g_cuCtxSetCurrent(g_cudaContext);
    const uint64_t copy_t0 = debug ? DStorageLoaderTimeUS() : 0;
    for (int i = 0; i < request_count; ++i) {
        const BatchSegment & seg = segments[(size_t) i];
        uint64_t copy_size = (requests[i].uncompressed_size > requests[i].size) ? requests[i].uncompressed_size : requests[i].size;
        CUresult cr;
        if (loader->cudaStream && g_cuMemcpyDtoDAsync) {
            cr = g_cuMemcpyDtoDAsync(
                (CUdeviceptr) requests[i].cuda_dest_ptr,
                loader->staging.devPtr + seg.staging_offset + seg.align_offset,
                (size_t) copy_size,
                loader->cudaStream);
        } else {
            cr = g_cuMemcpyDtoD(
                (CUdeviceptr) requests[i].cuda_dest_ptr,
                loader->staging.devPtr + seg.staging_offset + seg.align_offset,
                (size_t) copy_size);
        }
        if (cr != CUDA_SUCCESS) {
            g_lastHR = (HRESULT)(0xCD000000 | (cr & 0xFFFF));
            return -1;
        }
    }
    const uint64_t copy_t1 = debug ? DStorageLoaderTimeUS() : 0;

    if (loader->cudaStream && loader->copyEvent && g_cuEventRecord) {
        g_cuEventRecord(loader->copyEvent, loader->cudaStream);
    } else {
        if (g_cuCtxSynchronize) {
            g_cuCtxSynchronize();
        }
    }

    if (debug) {
        std::fprintf(stderr,
            "DStorage DEBUG loader:batch requests=%d staging_mib=%.2f staging_us=%" PRIu64 " open_enqueue_us=%" PRIu64 " submit_wait_us=%" PRIu64 " copy_us=%" PRIu64 " total_us=%" PRIu64 "\n",
            request_count,
            staging_size / 1024.0 / 1024.0,
            after_staging_us - total_t0,
            submit_t0 - after_staging_us,
            submit_t1 - submit_t0,
            copy_t1 - copy_t0,
            copy_t1 - total_t0);
        std::fflush(stderr);
    }

    return 0;
}

void* ds_loader_host_alloc(uint64_t size, int* out_is_pinned) {
    if (out_is_pinned) {
        *out_is_pinned = 0;
    }
    if (size == 0) {
        return nullptr;
    }

    if (EnsureCudaLoaded() && g_cuMemHostAlloc) {
        void* ptr = nullptr;
        CUresult cr = g_cuMemHostAlloc(&ptr, (size_t) size, 0);
        if (cr == CUDA_SUCCESS && ptr != nullptr) {
            if (out_is_pinned) {
                *out_is_pinned = 1;
            }
            return ptr;
        }
    }

    return _aligned_malloc((size_t) size, 4096);
}

void ds_loader_host_free(void* ptr, int is_pinned) {
    if (ptr == nullptr) {
        return;
    }
    if (is_pinned && g_cudaInitialized && g_cuMemFreeHost) {
        g_cuCtxSetCurrent(g_cudaContext);
        g_cuMemFreeHost(ptr);
        return;
    }
    _aligned_free(ptr);
}

int ds_loader_host_to_cuda_batch(
    DSLoaderHandle loader,
    const DSLoaderHostToCudaRequest* requests,
    int request_count
) {
    if (!loader || !requests || request_count <= 0) return -1;
    if (!EnsureCudaLoaded()) { g_lastHR = E_FAIL; return -1; }
    const bool debug = DStorageLoaderDebugEnabled();
    const uint64_t t0 = debug ? DStorageLoaderTimeUS() : 0;

    g_cuCtxSetCurrent(g_cudaContext);
    uint64_t total_bytes = 0;
    for (int i = 0; i < request_count; ++i) {
        const DSLoaderHostToCudaRequest & req = requests[i];
        if (req.host_src_ptr == nullptr || req.size == 0 || req.cuda_dest_ptr == 0) {
            return -1;
        }
        CUresult cr;
        if (loader->cudaStream && g_cuMemcpyHtoDAsync) {
            cr = g_cuMemcpyHtoDAsync(
                (CUdeviceptr) req.cuda_dest_ptr,
                req.host_src_ptr,
                (size_t) req.size,
                loader->cudaStream);
        } else {
            cr = g_cuMemcpyHtoD(
                (CUdeviceptr) req.cuda_dest_ptr,
                req.host_src_ptr,
                (size_t) req.size);
        }
        if (cr != CUDA_SUCCESS) {
            g_lastHR = (HRESULT)(0xCF000000 | (cr & 0xFFFF));
            return -1;
        }
        total_bytes += req.size;
    }

    if (loader->cudaStream && loader->copyEvent && g_cuEventRecord) {
        g_cuEventRecord(loader->copyEvent, loader->cudaStream);
    } else {
        if (g_cuCtxSynchronize) {
            g_cuCtxSynchronize();
        }
    }

    if (debug) {
        const uint64_t t1 = DStorageLoaderTimeUS();
        std::fprintf(stderr,
            "DStorage DEBUG loader:pinned_batch requests=%d bytes_mib=%.2f copy_us=%" PRIu64 "\n",
            request_count,
            total_bytes / 1024.0 / 1024.0,
            t1 - t0);
        std::fflush(stderr);
    }

    return 0;
}

int ds_loader_get_io_stats(DSLoaderHandle, DSLoaderIoStats* out_stats) {
    if (!out_stats) {
        return -1;
    }
    *out_stats = {};
    return 0;
}

int ds_loader_cuda_to_host_batch(
    DSLoaderHandle,
    const DSLoaderCudaToHostRequest* requests,
    int request_count
) {
    if (!requests || request_count < 0 || !EnsureCudaLoaded()) return -1;
    g_cuCtxSetCurrent(g_cudaContext);
    for (int i = 0; i < request_count; ++i) {
        const DSLoaderCudaToHostRequest & req = requests[i];
        CUresult cr = g_cuMemcpyDtoH(
            req.host_dest_ptr,
            (CUdeviceptr) req.cuda_src_ptr,
            (size_t) req.size);
        if (cr != CUDA_SUCCESS) {
            g_lastHR = (HRESULT)(0xCE000000 | (cr & 0xFFFF));
            return -1;
        }
    }
    return 0;
}

// --- CUDA memory allocation (for testing) ---
uint64_t ds_loader_cuda_alloc(uint64_t size) {
    if (!EnsureCudaLoaded() || size == 0) return 0;
    g_cuCtxSetCurrent(g_cudaContext);
    CUdeviceptr ptr = 0;
    CUresult cr = g_cuMemAlloc(&ptr, (size_t)size);
    if (cr != CUDA_SUCCESS) return 0;

    // Dynamically load cuMemsetD8 to zero out the memory
    typedef int (*PFN_cuMemsetD8)(unsigned long long dstDevice, unsigned char uc, size_t N);
    PFN_cuMemsetD8 pfn_cuMemsetD8 = (PFN_cuMemsetD8)GetProcAddress((HMODULE)g_nvcudaModule, "cuMemsetD8");
    if (pfn_cuMemsetD8) {
        pfn_cuMemsetD8((unsigned long long)ptr, 0, (size_t)size);
    }

    return (uint64_t)ptr;
}

void ds_loader_cuda_free(uint64_t ptr) {
    if (!g_cudaInitialized || ptr == 0) return;
    g_cuCtxSetCurrent(g_cudaContext);
    g_cuMemFree((CUdeviceptr)ptr);
}

// Copy from a raw CUDA device pointer to host memory (for testing StreamToCuda)
int ds_loader_cuda_dtoh(uint64_t src_cuda_ptr, void* dest, uint64_t size) {
    if (!g_cudaInitialized || src_cuda_ptr == 0 || !dest || size == 0) return -1;
    g_cuCtxSetCurrent(g_cudaContext);
    CUresult cr = g_cuMemcpyDtoH(dest, (CUdeviceptr)src_cuda_ptr, (size_t)size);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCE000000 | (cr & 0xFFFF));
        return -1;
    }
    return 0;
}

int ds_loader_cuda_dtod(uint64_t dst_cuda_ptr, uint64_t src_cuda_ptr, uint64_t size) {
    if (!g_cudaInitialized || dst_cuda_ptr == 0 || src_cuda_ptr == 0 || size == 0) return -1;
    g_cuCtxSetCurrent(g_cudaContext);
    CUresult cr = g_cuMemcpyDtoD((CUdeviceptr) dst_cuda_ptr, (CUdeviceptr) src_cuda_ptr, (size_t) size);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCE000000 | (cr & 0xFFFF));
        return -1;
    }
    return 0;
}

void ds_loader_cuda_destroy(CUDAInteropHandle interop) {
    if (!interop) return;

    if (g_cudaInitialized) {
        g_cuCtxSetCurrent(g_cudaContext);
        if (interop->devPtr) {
            g_cuMemFree(interop->devPtr);
        }
        if (interop->extMem) {
            g_cuDestroyExternalMemory(interop->extMem);
        }
    }
    if (interop->sharedHandle) {
        CloseHandle(interop->sharedHandle);
    }

    delete interop;
}

int ds_loader_cuda_wait_event(DSLoaderHandle loader) {
    if (!loader) return -1;
    if (!g_cudaInitialized) return 0;
    if (!loader->copyEvent || !g_cuEventSynchronize) return 0;

    g_cuCtxSetCurrent(g_cudaContext);
    CUresult cr = g_cuEventSynchronize(loader->copyEvent);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCE000000 | (cr & 0xFFFF));
        return -1;
    }
    return 0;
}

extern "C" {
DS_API int ds_loader_compress_buffer(const void* src, uint64_t src_size, void* dst, uint64_t dst_size, uint64_t* out_compressed_size) {
    if (!EnsureDStorageLoaded() || !g_DStorageCreateCompressionCodec) return -1;
    ComPtr<IDStorageCompressionCodec> codec;
    HRESULT hr = g_DStorageCreateCompressionCodec(DSTORAGE_COMPRESSION_FORMAT_GDEFLATE, 0, IID_PPV_ARGS(&codec));
    if (FAILED(hr)) return -1;
    size_t comp_size = 0;
    hr = codec->CompressBuffer(src, src_size, DSTORAGE_COMPRESSION_BEST_RATIO, dst, dst_size, &comp_size);
    if (FAILED(hr)) return -1;
    if (out_compressed_size) {
        *out_compressed_size = comp_size;
    }
    return 0;
}
}

} // extern "C"

#else // !_WIN32

#include "dstorage_loader.h"
#include <stddef.h>

extern "C" {

int ds_loader_available() { return 0; }
int32_t ds_loader_get_hresult() { return 0; }
DSLoaderHandle ds_loader_create() { return NULL; }
void ds_loader_destroy(DSLoaderHandle loader) {}
void* ds_loader_create_gpu_buffer(DSLoaderHandle loader, uint64_t size) { return NULL; }
void ds_loader_destroy_gpu_buffer(void* gpu_buffer) {}
int ds_loader_read(DSLoaderHandle loader, const wchar_t* file_path,
                   uint64_t file_offset, uint64_t size, void* gpu_buffer) { return -1; }
int ds_loader_read_chunked(DSLoaderHandle loader, const wchar_t* file_path,
                            uint64_t file_offset, uint64_t total_size, void* gpu_buffer) { return -1; }
int ds_loader_read_to_memory(DSLoaderHandle loader, const wchar_t* file_path,
                              uint64_t file_offset, uint64_t size, void* dest_memory) { return -1; }
int ds_loader_open_file(DSLoaderHandle loader, const wchar_t* file_path) { return -1; }
void ds_loader_close_file(DSLoaderHandle loader) {}
int ds_loader_enqueue_read(DSLoaderHandle loader, uint64_t file_offset,
                            uint64_t size, void* gpu_buffer, uint64_t buffer_offset) { return -1; }
int ds_loader_submit_and_wait(DSLoaderHandle loader) { return -1; }
int ds_loader_submit(DSLoaderHandle loader) { return -1; }
int ds_loader_is_complete(DSLoaderHandle loader) { return 1; }
int ds_loader_wait_complete(DSLoaderHandle loader) { return -1; }
int ds_loader_gpu_readback(DSLoaderHandle loader, void* gpu_buffer,
                            uint64_t size, void* dest_memory) { return -1; }
int ds_loader_debug_shared(DSLoaderHandle loader) { return -1; }
int ds_loader_cuda_available() { return 0; }
void* ds_loader_create_shared_gpu_buffer(DSLoaderHandle loader, uint64_t size) { return NULL; }
CUDAInteropHandle ds_loader_export_to_cuda(DSLoaderHandle loader,
                                            void* shared_gpu_buffer, uint64_t size) { return NULL; }
uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle interop) { return 0; }
int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle interop, void* dest, uint64_t size) { return -1; }
void ds_loader_cuda_destroy(CUDAInteropHandle interop) {}
int ds_loader_stream_to_cuda(DSLoaderHandle loader, const wchar_t* file_path,
                              uint64_t file_offset, uint64_t size, uint64_t cuda_dest_ptr) { return -1; }
int ds_loader_stream_to_cuda_batch(DSLoaderHandle loader, const DSLoaderStreamRequest* requests,
                                    int request_count) { return -1; }
int ds_loader_get_io_stats(DSLoaderHandle loader, DSLoaderIoStats* out_stats) {
    if (out_stats) *out_stats = {};
    return out_stats ? 0 : -1;
}
void* ds_loader_host_alloc(uint64_t size, int* out_is_pinned) { if (out_is_pinned) *out_is_pinned = 0; return NULL; }
void ds_loader_host_free(void* ptr, int is_pinned) {}
int ds_loader_host_to_cuda_batch(DSLoaderHandle loader, const DSLoaderHostToCudaRequest* requests,
                                  int request_count) { return -1; }
int ds_loader_cuda_to_host_batch(DSLoaderHandle loader, const DSLoaderCudaToHostRequest* requests,
                                  int request_count) { return -1; }
uint64_t ds_loader_cuda_alloc(uint64_t size) { return 0; }
void ds_loader_cuda_free(uint64_t ptr) {}
int ds_loader_cuda_dtoh(uint64_t src_cuda_ptr, void* dest, uint64_t size) { return -1; }
int ds_loader_cuda_dtod(uint64_t dst_cuda_ptr, uint64_t src_cuda_ptr, uint64_t size) { return -1; }
int ds_loader_cuda_wait_event(DSLoaderHandle loader) { return 0; }
int ds_loader_compress_buffer(const void* src, uint64_t src_size, void* dst, uint64_t dst_size, uint64_t* out_compressed_size) { return -1; }

} // extern "C"

#endif // _WIN32
