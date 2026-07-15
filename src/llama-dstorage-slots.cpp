#include "llama-dstorage-slots.h"

#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cwchar>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(_WIN32) || defined(__linux__)
#include "../ggml/src/ggml-cuda/dstorage_loader.h"
#define LLAMA_DSTORAGE_HAS_BACKEND 1
#else
#define LLAMA_DSTORAGE_HAS_BACKEND 0
#endif

static int64_t llama_dstorage_now_us() {
    return ggml_time_us();
}

static bool llama_dstorage_summary_enabled();

// Aggregates per-step time across the whole run so the summary can show where the
// ensure-experts callback actually spends its wall time, without per-call logging.
struct EnsureStepProfile {
    std::mutex mutex;
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> steps; // key -> {us, calls}

    void record(const char * scope, const char * step, int64_t us) {
        if (us < 0) {
            us = 0;
        }
        std::string key = std::string(scope) + ":" + step;
        std::lock_guard<std::mutex> lock(mutex);
        auto & e = steps[key];
        e.first += uint64_t(us);
        e.second += 1;
    }
};
static EnsureStepProfile g_ensure_step_profile;

static void llama_dstorage_trace_us(
        const char * scope,
        int layer_idx,
        const char * step,
        int64_t t0,
        int64_t t1) {
    if (llama_dstorage_summary_enabled()) {
        g_ensure_step_profile.record(scope, step, t1 - t0);
    }
    if (llama_dstorage_debug_enabled()) {
        std::fprintf(stderr, "DStorage TRACE %s layer=%d step=%s us=%lld\n",
                scope, layer_idx, step, (long long) (t1 - t0));
        std::fflush(stderr);
    }
}

static bool llama_dstorage_summary_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_SUMMARY");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool llama_dstorage_timeline_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_TIMELINE_TRACE");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool llama_dstorage_align_pool_base_4k() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_ALIGN_POOL_BASE_4K");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool llama_dstorage_align_pool_stride_4k() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_ALIGN_POOL_STRIDE_4K_UNSAFE");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool llama_dstorage_fused_expert_pools() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_FUSED_EXPERT_POOLS");
        const char * unsafe = std::getenv("LLAMA_DSTORAGE_FUSED_EXPERT_POOLS_UNSAFE");
        return value != nullptr &&
               value[0] != '\0' &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               unsafe != nullptr &&
               unsafe[0] != '\0' &&
               std::strcmp(unsafe, "0") != 0 &&
               std::strcmp(unsafe, "false") != 0 &&
               std::strcmp(unsafe, "FALSE") != 0;
    }();
    return enabled;
}

struct llama_dstorage_sidecar_entry {
    std::string path;
    uint64_t offset = 0;
    uint64_t total_size = 0;
    int n_experts = 0;
    bool expert_major = false;
    uint64_t record_stride = 0;
    uint64_t tensor_offset = 0;
};

static std::string llama_dstorage_sidecar_key(int layer_idx, const char * tensor_name) {
    return std::to_string(layer_idx) + ":" + (tensor_name != nullptr ? tensor_name : "");
}

static const std::unordered_map<std::string, llama_dstorage_sidecar_entry> & llama_dstorage_sidecar_manifest() {
    static const std::unordered_map<std::string, llama_dstorage_sidecar_entry> manifest = [] {
        std::unordered_map<std::string, llama_dstorage_sidecar_entry> out;
        const char * path_env = std::getenv("LLAMA_MOE_EXPERT_SIDECAR_MANIFEST");
        if (path_env == nullptr || path_env[0] == '\0') {
            return out;
        }

        std::ifstream file(path_env);
        if (!file) {
            LLAMA_LOG_ERROR("DirectStorage: failed to open sidecar manifest '%s'\n", path_env);
            return out;
        }

        std::string line;
        uint64_t line_no = 0;
        while (std::getline(file, line)) {
            line_no++;
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream iss(line);
            int layer = -1;
            std::string tensor_name;
            std::string sidecar_path;
            uint64_t offset = 0;
            uint64_t total_size = 0;
            int n_experts = 0;
            if (!(iss >> layer >> tensor_name >> sidecar_path >> offset >> total_size >> n_experts)) {
                LLAMA_LOG_WARN(
                        "DirectStorage: ignoring malformed sidecar manifest line %" PRIu64 " in '%s'\n",
                        line_no, path_env);
                continue;
            }
            llama_dstorage_sidecar_entry entry;
            entry.path = sidecar_path;
            entry.offset = offset;
            entry.total_size = total_size;
            entry.n_experts = n_experts;
            std::string layout;
            if (iss >> layout) {
                if (layout == "expert_major") {
                    entry.expert_major = true;
                    if (!(iss >> entry.record_stride >> entry.tensor_offset)) {
                        LLAMA_LOG_WARN(
                                "DirectStorage: ignoring malformed expert-major sidecar fields on line %" PRIu64 " in '%s'\n",
                                line_no, path_env);
                        continue;
                    }
                } else if (layout != "tensor_major") {
                    LLAMA_LOG_WARN(
                            "DirectStorage: ignoring unknown sidecar layout '%s' on line %" PRIu64 " in '%s'\n",
                            layout.c_str(), line_no, path_env);
                    continue;
                }
            }
            out[llama_dstorage_sidecar_key(layer, tensor_name.c_str())] = entry;
        }

        LLAMA_LOG_INFO(
                "DirectStorage: loaded %zu sidecar expert tensor entries from '%s'\n",
                out.size(), path_env);
        return out;
    }();
    return manifest;
}

static uint64_t llama_dstorage_pool_visible_ptr(uint64_t alloc_ptr) {
    return llama_dstorage_align_pool_base_4k()
            ? ((alloc_ptr + 4095ULL) & ~4095ULL)
            : alloc_ptr;
}

static uint64_t llama_dstorage_pool_alloc_size(uint64_t pool_size) {
    return llama_dstorage_align_pool_base_4k() ? pool_size + 4095ULL : pool_size;
}

static uint64_t llama_dstorage_expert_file_offset(const ExpertTensorInfo & info, int expert_idx) {
    if (info.sidecar_expert_major) {
        return info.file_offset +
               uint64_t(expert_idx) * info.sidecar_record_stride +
               info.sidecar_tensor_offset;
    }
    return info.file_offset + uint64_t(expert_idx) * info.expert_stride;
}

static bool llama_dstorage_hybrid_arc_requested() {
    const char * value = std::getenv("LLAMA_DSTORAGE_CACHE_POLICY");
    return value != nullptr && std::strcmp(value, "hybrid_arc") == 0;
}

static const char * llama_dstorage_routing_trace_path() {
    const char * value = std::getenv("LLAMA_DSTORAGE_ROUTING_TRACE");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

static bool llama_dstorage_predictive_prefetch_enabled() {
    const char * value = std::getenv("LLAMA_DSTORAGE_PREDICTIVE_PREFETCH");
    return value != nullptr &&
           std::strcmp(value, "") != 0 &&
           std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

static double llama_dstorage_prefetch_min_confidence() {
    const char * env = std::getenv("LLAMA_DSTORAGE_PREFETCH_MIN_CONF");
    if (env == nullptr || env[0] == '\0') {
        return llama_dstorage_predictive_prefetch_enabled() ? 0.0625 : 0.125;
    }
    char * end = nullptr;
    const double value = std::strtod(env, &end);
    return end != env ? std::max(0.0, std::min(value, 1.0)) : 0.125;
}

static int llama_dstorage_prefetch_min_observations() {
    const char * env = std::getenv("LLAMA_DSTORAGE_PREFETCH_MIN_OBSERVATIONS");
    if (env == nullptr || env[0] == '\0') {
        return llama_dstorage_predictive_prefetch_enabled() ? 1 : 2;
    }
    return std::max(1, std::min(std::atoi(env), 1024));
}

static bool llama_dstorage_high_confidence_prefetch_enabled() {
    const char * value = std::getenv("LLAMA_DSTORAGE_PREFETCH_POLICY");
    return value != nullptr && std::strcmp(value, "high_confidence") == 0;
}

static int llama_dstorage_prefetch_max_experts_per_layer(int fallback) {
    const char * env = std::getenv("LLAMA_DSTORAGE_PREFETCH_MAX_EXPERTS_PER_LAYER");
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    return std::max(1, std::min(std::atoi(env), std::max(1, fallback)));
}

static int llama_dstorage_predictive_prefetch_max_submissions() {
    const char * env = std::getenv("LLAMA_DSTORAGE_PREDICTIVE_PREFETCH_MAX_SUBMISSIONS");
    if (env == nullptr || env[0] == '\0') {
        return 2;
    }
    return std::max(1, std::min(std::atoi(env), 16));
}

static bool llama_dstorage_decode_hot_cache_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_DECODE_HOT_CACHE");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static int llama_dstorage_decode_hot_cache_slots() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_DSTORAGE_DECODE_HOT_CACHE_SLOTS");
        if (env == nullptr || env[0] == '\0') {
            return 256;
        }
        return std::max(1, std::min(std::atoi(env), 4096));
    }();
    return value;
}

static int llama_dstorage_decode_hot_min_hits() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_DSTORAGE_DECODE_HOT_MIN_HITS");
        if (env == nullptr || env[0] == '\0') {
            return 2;
        }
        return std::max(1, std::min(std::atoi(env), 1024));
    }();
    return value;
}

static int llama_dstorage_decode_hot_rebuild_interval() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_DSTORAGE_DECODE_HOT_REBUILD_INTERVAL");
        if (env == nullptr || env[0] == '\0') {
            return 64;
        }
        return std::max(1, std::min(std::atoi(env), 4096));
    }();
    return value;
}

static bool llama_dstorage_decode_hot_prefill_like_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_DECODE_HOT_PREFILL_LIKE");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool llama_dstorage_pinned_prefill_only_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_MOE_PINNED_PREFILL_ONLY");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static uint64_t llama_dstorage_pinned_prefill_only_min_pinned_bytes(uint64_t pinned_budget_bytes) {
    const char * env = std::getenv("LLAMA_MOE_PINNED_PREFILL_ONLY_MIN_PINNED_MIB");
    if (env != nullptr && env[0] != '\0') {
        const int parsed = std::atoi(env);
        return uint64_t(std::max(0, parsed)) * 1024ull * 1024ull;
    }
    return pinned_budget_bytes * 4ull;
}

static uint32_t llama_dstorage_pinned_admit_max_per_call() {
    static const uint32_t value = [] {
        const char * env = std::getenv("LLAMA_MOE_PINNED_CACHE_ADMIT_MAX_PER_CALL");
        if (env == nullptr || env[0] == '\0') {
            return 0u;
        }
        const int parsed = std::atoi(env);
        return uint32_t(std::max(0, std::min(parsed, 1024)));
    }();
    return value;
}

static uint64_t llama_dstorage_pinned_admit_max_pending_bytes() {
    static const uint64_t value = [] {
        const char * env = std::getenv("LLAMA_MOE_PINNED_CACHE_ADMIT_MAX_PENDING_MIB");
        if (env == nullptr || env[0] == '\0') {
            return 256ull * 1024ull * 1024ull;
        }
        const int parsed = std::atoi(env);
        return uint64_t(std::max(0, std::min(parsed, 65536))) * 1024ull * 1024ull;
    }();
    return value;
}

static bool llama_dstorage_mtp_decode_phase_enabled() {
    const char * value = std::getenv("LLAMA_DSTORAGE_MTP_DECODE_PHASE");
    return value != nullptr &&
           std::strcmp(value, "") != 0 &&
           std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

static std::vector<int> llama_dstorage_parse_static_pinned_layers(int n_layers) {
    std::vector<int> layers;
    const char * env = std::getenv("LLAMA_MOE_PINNED_STATIC_LAYERS");
    if (env == nullptr || env[0] == '\0') {
        return layers;
    }

    const char * p = env;
    while (*p != '\0') {
        while (*p == ',' || std::isspace(static_cast<unsigned char>(*p))) {
            ++p;
        }
        if (*p == '\0') {
            break;
        }

        char * end = nullptr;
        long first = std::strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        p = end;

        long last = first;
        if (*p == '-') {
            ++p;
            last = std::strtol(p, &end, 10);
            if (end == p) {
                break;
            }
            p = end;
        }
        if (last < first) {
            std::swap(last, first);
        }
        for (long layer = first; layer <= last; ++layer) {
            if (layer >= 0 && layer < n_layers) {
                const int value = int(layer);
                if (std::find(layers.begin(), layers.end(), value) == layers.end()) {
                    layers.push_back(value);
                }
            }
        }
    }

    std::sort(layers.begin(), layers.end());
    return layers;
}

static std::vector<std::pair<int, int>> llama_dstorage_parse_static_pinned_experts(
        int n_layers,
        int n_experts) {
    std::vector<std::pair<int, int>> experts;
    const char * path = std::getenv("LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE");
    if (path == nullptr || path[0] == '\0') {
        return experts;
    }

    FILE * file = std::fopen(path, "r");
    if (file == nullptr) {
        LLAMA_LOG_WARN(
                "DirectStorage: failed to open static pinned expert file '%s': %s\n",
                path,
                std::strerror(errno));
        return experts;
    }

    std::unordered_set<uint64_t> seen;
    char line[512];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        char * p = line;
        while (std::isspace(static_cast<unsigned char>(*p))) {
            ++p;
        }
        if (*p == '\0' || *p == '#') {
            continue;
        }

        char * end = nullptr;
        const long layer = std::strtol(p, &end, 10);
        if (end == p) {
            continue;
        }
        p = end;
        while (*p == ',' || *p == ':' || std::isspace(static_cast<unsigned char>(*p))) {
            ++p;
        }
        const long expert = std::strtol(p, &end, 10);
        if (end == p || layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) {
            continue;
        }

        const uint64_t key = (uint64_t(uint32_t(layer)) << 32) | uint32_t(expert);
        if (seen.insert(key).second) {
            experts.emplace_back(int(layer), int(expert));
        }
    }
    std::fclose(file);
    return experts;
}

#if defined(__linux__)
static bool llama_dstorage_pread_full(int fd, uint64_t offset, void * dst, uint64_t size) {
    char * out = reinterpret_cast<char *>(dst);
    uint64_t done = 0;
    while (done < size) {
        const size_t chunk = size_t(std::min<uint64_t>(size - done, 1024ull * 1024ull * 1024ull));
        const ssize_t nread = pread(fd, out + done, chunk, off_t(offset + done));
        if (nread <= 0) {
            return false;
        }
        done += uint64_t(nread);
    }
    return true;
}
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string DStorageSlotManager::make_tensor_key(int layer_idx, const char * tensor_name) {
    return std::to_string(layer_idx) + ":" + tensor_name;
}

std::string DStorageSlotManager::get_tensor_type_key(const std::string & tensor_name) {
    // "blk.5.ffn_gate_up_exps.weight" -> "ffn_gate_up_exps.weight"
    auto dot1 = tensor_name.find('.');
    if (dot1 == std::string::npos) { return tensor_name; }
    auto dot2 = tensor_name.find('.', dot1 + 1);
    if (dot2 == std::string::npos) { return tensor_name; }
    return tensor_name.substr(dot2 + 1);
}

int DStorageSlotManager::parse_layer_idx(const std::string & tensor_name) {
    if (tensor_name.rfind("blk.", 0) != 0) {
        return -1;
    }
    const size_t begin = 4;
    const size_t end = tensor_name.find('.', begin);
    if (end == std::string::npos || end == begin) {
        return -1;
    }
    int layer = 0;
    for (size_t i = begin; i < end; ++i) {
        if (tensor_name[i] < '0' || tensor_name[i] > '9') {
            return -1;
        }
        layer = layer * 10 + (tensor_name[i] - '0');
    }
    return layer;
}

std::string DStorageSlotManager::make_pool_key(int pool_group, const std::string & type_key) {
    return std::to_string(pool_group) + ":" + type_key;
}

int DStorageSlotManager::pool_group_for_layer(int layer_idx) const {
    const auto it = layer_pool_groups_.find(layer_idx);
    return it == layer_pool_groups_.end() ? 0 : it->second;
}

std::string DStorageSlotManager::pool_key_for_layer(int layer_idx, const std::string & tensor_name) const {
    return make_pool_key(pool_group_for_layer(layer_idx), get_tensor_type_key(tensor_name));
}

bool DStorageSlotManager::slot_compatible_with_layer(int slot_idx, int layer_idx) const {
    if (slot_idx < 0 || slot_idx >= int(slots_.size())) {
        return false;
    }
    return slots_[slot_idx].pool_group == pool_group_for_layer(layer_idx);
}

uint64_t DStorageSlotManager::get_slot_stride(const std::string & tensor_name) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    const int layer_idx = parse_layer_idx(tensor_name);
    const std::string pool_key = layer_idx >= 0
            ? pool_key_for_layer(layer_idx, tensor_name)
            : make_pool_key(0, get_tensor_type_key(tensor_name));
    auto it = type_pools_.find(pool_key);
    if (it != type_pools_.end()) {
        return it->second.slot_size;
    }
    return 0;
}

int DStorageSlotManager::get_slot_count(const std::string & tensor_name) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    const int layer_idx = parse_layer_idx(tensor_name);
    const int group = layer_idx >= 0 ? pool_group_for_layer(layer_idx) : 0;
    if (group >= 0 && group < int(pool_group_slot_counts_.size())) {
        return pool_group_slot_counts_[group];
    }
    return n_slots_;
}

uint64_t DStorageSlotManager::get_prefill_workspace_slot_stride(const std::string & tensor_name) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    auto it = type_strides_.find(get_tensor_type_key(tensor_name));
    return it != type_strides_.end() ? it->second : 0;
}

int DStorageSlotManager::get_prefill_workspace_slot_count() const {
    return n_experts_;
}

static int llama_dstorage_max_coalesced_experts() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_DSTORAGE_COALESCE_EXPERTS");
        if (env == nullptr || env[0] == '\0') {
#if defined(__linux__)
            return 8;
#else
            return 1;
#endif
        }

        const int parsed = std::atoi(env);
        return std::max(1, std::min(parsed, 128));
    }();
    return value;
}

static bool llama_dstorage_contiguous_load_slots() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_CONTIGUOUS_LOAD_SLOTS");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static uint64_t llama_dstorage_max_batch_staging_bytes() {
    const char * env = std::getenv("LLAMA_DSTORAGE_MAX_BATCH_STAGING_MIB");
    if (env != nullptr) {
        char * end = nullptr;
        const unsigned long long mib = std::strtoull(env, &end, 10);
        if (end != env && *end == '\0' && mib > 0) {
            return uint64_t(mib) * 1024ull * 1024ull;
        }
    }
    return 32ull * 1024ull * 1024ull;
}

static bool llama_dstorage_sort_batch_by_offset() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_SORT_BATCH_BY_OFFSET");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool llama_dstorage_bundle_contiguous_reads() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_BUNDLE_CONTIGUOUS_READS");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static uint64_t llama_dstorage_bundle_max_read_bytes() {
    const char * env = std::getenv("LLAMA_DSTORAGE_BUNDLE_MAX_READ_KIB");
    if (env != nullptr) {
        char * end = nullptr;
        const unsigned long long kib = std::strtoull(env, &end, 10);
        if (end != env && *end == '\0' && kib > 0) {
            return uint64_t(kib) * 1024ull;
        }
    }
    return 0;
}

static uint64_t llama_dstorage_bundle_stripe_bytes() {
    const char * env = std::getenv("LLAMA_DSTORAGE_BUNDLE_STRIPE_KIB");
    if (env != nullptr) {
        char * end = nullptr;
        const unsigned long long kib = std::strtoull(env, &end, 10);
        if (end != env && *end == '\0' && kib > 0) {
            return ((uint64_t(kib) * 1024ull) / 4096ull) * 4096ull;
        }
    }
    return 0;
}

static bool llama_dstorage_global_prefetch_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_GLOBAL_PREFETCH");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static bool llama_dstorage_relax_prefetch_layer_guard() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_RELAX_PREFETCH_LAYER_GUARD");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

static int llama_dstorage_force_prefetch_max_pending() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_DSTORAGE_FORCE_PREFETCH_MAX_PENDING");
        if (env == nullptr || env[0] == '\0') {
            return 1;
        }
        return std::max(1, std::min(std::atoi(env), 16));
    }();
    return value;
}

static int llama_dstorage_max_prefetch_distance() {
    static const int value = [] {
        const char * env = std::getenv("LLAMA_DSTORAGE_PREFETCH_MAX_DISTANCE");
        if (env == nullptr || env[0] == '\0') {
            return 4;
        }
        return std::max(1, std::min(std::atoi(env), 32));
    }();
    return value;
}

uint64_t DStorageSlotManager::make_expert_key(int layer_idx, int expert_idx) {
    return (uint64_t(uint32_t(layer_idx)) << 32) | uint32_t(expert_idx);
}

uint32_t DStorageSlotManager::pinned_admit_min_hits() const {
    static const uint32_t value = [] {
        const char * env = std::getenv("LLAMA_MOE_PINNED_CACHE_ADMIT_MIN_HITS");
        if (env == nullptr || env[0] == '\0') {
            return 8u;
        }
        const int parsed = std::atoi(env);
        return uint32_t(std::max(1, std::min(parsed, 1024)));
    }();
    return value;
}

bool DStorageSlotManager::pinned_cache_enabled() const {
    return pinned_budget_bytes_ > 0 && ds_loader_ != nullptr;
}

bool DStorageSlotManager::static_pinned_entry(uint64_t expert_key) const {
    return static_pinned_experts_.find(expert_key) != static_pinned_experts_.end();
}

void DStorageSlotManager::set_speculative_prefetch_enabled(bool enabled) {
    speculative_prefetch_enabled_ = enabled;
}

int DStorageSlotManager::speculative_slot_count() const {
    if (!speculative_prefetch_enabled_ || n_slots_ < 64 || n_expert_used_ <= 0) {
        return 0;
    }
    const char * env = std::getenv("LLAMA_DSTORAGE_SPECULATIVE_SLOTS");
    if (env != nullptr && env[0] != '\0') {
        return calculate_speculative_slot_count(n_slots_, n_expert_used_);
    }
    return std::min(n_slots_ / 4, std::max(32, n_expert_used_ * 8));
}

bool DStorageSlotManager::hybrid_arc_enabled() const {
    return hybrid_arc_policy_;
}

int DStorageSlotManager::layer_target_slots(int layer_idx) const {
    if (n_slots_ <= 0 || n_layers_ <= 0 || layer_idx < 0) {
        return 0;
    }
    const int shallow_layers = std::max(1, n_layers_ / 5);
    const int total_weight = n_layers_ + shallow_layers;
    const int layer_weight = layer_idx < shallow_layers ? 2 : 1;
    return std::max(n_expert_used_, (n_slots_ * layer_weight + total_weight - 1) / total_weight);
}

uint64_t DStorageSlotManager::expert_reload_bytes(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= int(layer_reload_bytes_.size())) {
        return 0;
    }
    return layer_reload_bytes_[layer_idx];
}

int DStorageSlotManager::contiguous_resident_neighbors(const ExpertSlot & slot) const {
    return slot.occupied ? slot.contiguous_neighbors : 0;
}

uint64_t DStorageSlotManager::decode_hot_score(uint64_t expert_key) const {
    const auto count_it = decode_hot_counts_.find(expert_key);
    const uint64_t count = count_it != decode_hot_counts_.end() ? count_it->second : 0;
    const auto miss_it = decode_hot_misses_.find(expert_key);
    const uint64_t misses = miss_it != decode_hot_misses_.end() ? miss_it->second : 0;
    const auto seen_it = decode_hot_last_seen_.find(expert_key);
    const uint64_t recency = seen_it != decode_hot_last_seen_.end() ? seen_it->second : 0;
    return count * 1000000ull + misses * 10000ull + recency;
}

bool DStorageSlotManager::decode_hot_protected(uint64_t expert_key) const {
    return llama_dstorage_decode_hot_cache_enabled() &&
           decode_hot_protected_.find(expert_key) != decode_hot_protected_.end();
}

bool DStorageSlotManager::decode_hot_protected(const ExpertSlot & slot) const {
    return slot.occupied &&
           slot.layer_idx >= 0 &&
           slot.expert_idx >= 0 &&
           decode_hot_protected(make_expert_key(slot.layer_idx, slot.expert_idx));
}

void DStorageSlotManager::rebuild_decode_hot_protected_set() {
    if (!llama_dstorage_decode_hot_cache_enabled()) {
        decode_hot_protected_.clear();
        return;
    }

    const int cap = std::min(
            std::max(1, llama_dstorage_decode_hot_cache_slots()),
            std::max(1, n_slots_ / 2));
    const uint32_t min_hits = uint32_t(llama_dstorage_decode_hot_min_hits());
    std::vector<std::pair<uint64_t, uint64_t>> ranked;
    ranked.reserve(decode_hot_counts_.size());
    for (const auto & entry : decode_hot_counts_) {
        if (entry.second < min_hits) {
            continue;
        }
        ranked.emplace_back(entry.first, decode_hot_score(entry.first));
    }
    std::sort(
            ranked.begin(),
            ranked.end(),
            [](const auto & lhs, const auto & rhs) {
                if (lhs.second != rhs.second) {
                    return lhs.second > rhs.second;
                }
                return lhs.first < rhs.first;
            });

    decode_hot_protected_.clear();
    for (const auto & entry : ranked) {
        if (int(decode_hot_protected_.size()) >= cap) {
            break;
        }
        decode_hot_protected_.insert(entry.first);
    }
    decode_hot_stats_.rebuilds++;
}

void DStorageSlotManager::reset_arc_state() {
    layer_arc_.assign(std::max(0, n_layers_), {});
    for (int layer = 0; layer < n_layers_; ++layer) {
        layer_arc_[layer].target_recent = layer_target_slots(layer) / 2;
    }
}

static void llama_dstorage_arc_erase(
        std::deque<int> & history,
        std::unordered_set<int> & history_set,
        int expert_idx) {
    history_set.erase(expert_idx);
    history.erase(std::remove(history.begin(), history.end(), expert_idx), history.end());
}

static void llama_dstorage_arc_add(
        std::deque<int> & history,
        std::unordered_set<int> & history_set,
        int expert_idx,
        int capacity) {
    llama_dstorage_arc_erase(history, history_set, expert_idx);
    history.push_back(expert_idx);
    history_set.insert(expert_idx);
    while (int(history.size()) > capacity) {
        history_set.erase(history.front());
        history.pop_front();
    }
}

void DStorageSlotManager::arc_record_hit(ExpertSlot & slot, dstorage_moe_phase phase) {
    if (!hybrid_arc_enabled() ||
            phase != dstorage_moe_phase::decode ||
            slot.layer_idx < 0 ||
            slot.layer_idx >= int(layer_arc_.size()) ||
            slot.arc_segment != dstorage_arc_segment::recent) {
        return;
    }
    LayerArcState & state = layer_arc_[slot.layer_idx];
    state.recent_residents = std::max(0, state.recent_residents - 1);
    state.frequent_residents++;
    slot.arc_segment = dstorage_arc_segment::frequent;
}

void DStorageSlotManager::arc_record_eviction(const ExpertSlot & slot) {
    if (!hybrid_arc_enabled()) {
        return;
    }
    if (slot.layer_idx >= 0 && slot.expert_idx >= 0) {
        for (int neighbor_expert : { slot.expert_idx - 1, slot.expert_idx + 1 }) {
            if (neighbor_expert < 0 || neighbor_expert >= n_experts_) {
                continue;
            }
            const auto neighbor = resident_experts_.find(
                    make_expert_key(slot.layer_idx, neighbor_expert));
            if (neighbor != resident_experts_.end() &&
                    neighbor->second >= 0 &&
                    neighbor->second < int(slots_.size())) {
                ExpertSlot & neighbor_slot = slots_[neighbor->second];
                neighbor_slot.contiguous_neighbors =
                        neighbor_slot.contiguous_neighbors > 0
                                ? neighbor_slot.contiguous_neighbors - 1
                                : 0;
            }
        }
        resident_experts_.erase(make_expert_key(slot.layer_idx, slot.expert_idx));
    }
    if (slot.layer_idx < 0 ||
            slot.layer_idx >= int(layer_arc_.size()) ||
            slot.arc_segment == dstorage_arc_segment::none) {
        return;
    }
    LayerArcState & state = layer_arc_[slot.layer_idx];
    const int history_capacity = std::max(2 * n_expert_used_, layer_target_slots(slot.layer_idx));
    if (slot.arc_segment == dstorage_arc_segment::recent) {
        state.recent_residents = std::max(0, state.recent_residents - 1);
        llama_dstorage_arc_add(
                state.recent_ghost,
                state.recent_ghost_set,
                slot.expert_idx,
                history_capacity);
    } else {
        state.frequent_residents = std::max(0, state.frequent_residents - 1);
        llama_dstorage_arc_add(
                state.frequent_ghost,
                state.frequent_ghost_set,
                slot.expert_idx,
                history_capacity);
    }
}

void DStorageSlotManager::arc_record_admission(ExpertSlot & slot, dstorage_moe_phase phase) {
    if (!hybrid_arc_enabled()) {
        slot.arc_segment = dstorage_arc_segment::none;
        slot.contiguous_neighbors = 0;
        return;
    }
    if (slot.layer_idx >= 0 && slot.expert_idx >= 0) {
        slot.contiguous_neighbors = 0;
        for (int neighbor_expert : { slot.expert_idx - 1, slot.expert_idx + 1 }) {
            if (neighbor_expert < 0 || neighbor_expert >= n_experts_) {
                continue;
            }
            const auto neighbor = resident_experts_.find(
                    make_expert_key(slot.layer_idx, neighbor_expert));
            if (neighbor != resident_experts_.end() &&
                    neighbor->second >= 0 &&
                    neighbor->second < int(slots_.size())) {
                slots_[neighbor->second].contiguous_neighbors++;
                slot.contiguous_neighbors++;
            }
        }
        resident_experts_[make_expert_key(slot.layer_idx, slot.expert_idx)] =
                int(&slot - slots_.data());
    }
    if (slot.layer_idx < 0 ||
            slot.layer_idx >= int(layer_arc_.size())) {
        slot.arc_segment = dstorage_arc_segment::none;
        return;
    }
    LayerArcState & state = layer_arc_[slot.layer_idx];
    const int capacity = layer_target_slots(slot.layer_idx);
    const bool decode_admission = phase == dstorage_moe_phase::decode;
    const bool recent_ghost_hit =
            decode_admission && state.recent_ghost_set.count(slot.expert_idx) != 0;
    const bool frequent_ghost_hit =
            decode_admission && state.frequent_ghost_set.count(slot.expert_idx) != 0;
    if (recent_ghost_hit) {
        const int delta = std::max(
                1,
                int(state.frequent_ghost_set.size()) /
                        std::max(1, int(state.recent_ghost_set.size())));
        state.target_recent = std::min(capacity, state.target_recent + delta);
        llama_dstorage_arc_erase(
                state.recent_ghost,
                state.recent_ghost_set,
                slot.expert_idx);
    } else if (frequent_ghost_hit) {
        const int delta = std::max(
                1,
                int(state.recent_ghost_set.size()) /
                        std::max(1, int(state.frequent_ghost_set.size())));
        state.target_recent = std::max(0, state.target_recent - delta);
        llama_dstorage_arc_erase(
                state.frequent_ghost,
                state.frequent_ghost_set,
                slot.expert_idx);
    }

    if (decode_admission || recent_ghost_hit || frequent_ghost_hit) {
        slot.arc_segment = dstorage_arc_segment::frequent;
        state.frequent_residents++;
    } else {
        slot.arc_segment = dstorage_arc_segment::recent;
        state.recent_residents++;
    }
}

void DStorageSlotManager::write_routing_trace(
        int layer_idx,
        dstorage_moe_phase phase,
        const std::vector<int32_t> & experts,
        const std::vector<int32_t> & misses,
        uint64_t stream_file_bytes,
        uint64_t transfer_us,
        uint64_t total_us) {
    if (routing_trace_file_ == nullptr) {
        return;
    }

    std::fprintf(
            routing_trace_file_,
            "{\"version\":1,\"sequence\":%" PRIu64 ",\"request\":%" PRIu64
            ",\"phase\":\"%s\",\"layer\":%d,\"slots\":%d,\"expert_bytes\":%" PRIu64
            ",\"stream_file_bytes\":%" PRIu64 ",\"transfer_us\":%" PRIu64
            ",\"total_us\":%" PRIu64 ",\"experts\":[",
            routing_trace_sequence_++,
            request_activation_stats_.generation(),
            llama_dstorage_phase_name(phase),
            layer_idx,
            n_slots_,
            expert_reload_bytes(layer_idx),
            stream_file_bytes,
            transfer_us,
            total_us);
    for (size_t i = 0; i < experts.size(); ++i) {
        std::fprintf(routing_trace_file_, "%s%d", i == 0 ? "" : ",", experts[i]);
    }
    std::fputs("],\"misses\":[", routing_trace_file_);
    for (size_t i = 0; i < misses.size(); ++i) {
        std::fprintf(routing_trace_file_, "%s%d", i == 0 ? "" : ",", misses[i]);
    }
    std::fputs("]}\n", routing_trace_file_);
    if (routing_trace_sequence_ % 256 == 0) {
        std::fflush(routing_trace_file_);
    }
}

bool DStorageSlotManager::pinned_admission_pending(uint64_t expert_key) const {
    for (const PendingPinnedAdmission & pending : pinned_admissions_) {
        for (uint64_t key : pending.expert_keys) {
            if (key == expert_key) {
                return true;
            }
        }
    }
    return false;
}

void DStorageSlotManager::collect_async_prefetches(bool wait_all) {
    std::vector<PendingAsyncPrefetch> to_wait;
    {
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        if (wait_all) {
            to_wait = std::move(async_prefetches_);
            async_prefetches_.clear();
        } else {
            for (auto it = async_prefetches_.begin(); it != async_prefetches_.end();) {
                if (it->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    to_wait.push_back(std::move(*it));
                    it = async_prefetches_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    if (to_wait.empty()) {
        return;
    }

    uint64_t ok_count = 0;
    uint64_t fail_count = 0;
    for (auto & pending : to_wait) {
        const bool ok = pending.future.get();
        if (ok) {
            ok_count++;
        } else {
            fail_count++;
        }
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG prefetch:async_complete layer=%d ids=%d layers=%zu ok=%d\n",
                pending.layer_idx, pending.n_selected, pending.affected_layers.size(), ok ? 1 : 0);
    }

    {
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        diagnostic_stats_.prefetch_completed_ok += ok_count;
        diagnostic_stats_.prefetch_completed_failed += fail_count;
    }
}

void DStorageSlotManager::collect_pinned_admissions(bool wait_all) {
#if LLAMA_DSTORAGE_HAS_BACKEND
    for (size_t i = 0; i < pinned_admissions_.size();) {
        PendingPinnedAdmission & pending = pinned_admissions_[i];
        const bool ready = wait_all ||
                pending.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        if (!ready) {
            ++i;
            continue;
        }

        std::vector<PinnedAdmissionResult> results = pending.future.get();
        if (pinned_pending_bytes_ >= pending.bytes) {
            pinned_pending_bytes_ -= pending.bytes;
        } else {
            pinned_pending_bytes_ = 0;
        }
        for (int slot_idx : pending.slot_indices) {
            if (slot_idx >= 0 && slot_idx < int(slots_.size())) {
                slots_[slot_idx].admission_pending = false;
            }
        }

        for (PinnedAdmissionResult & result : results) {
            if (result.ok && result.entry.host_ptr != nullptr &&
                    pinned_cache_.find(result.expert_key) == pinned_cache_.end()) {
                result.entry.last_used_tick = ++pinned_tick_;
                diagnostic_stats_.pinned_admit_completed++;
                diagnostic_stats_.pinned_admit_bytes += result.entry.bytes;
                diagnostic_stats_.pinned_admit_us += result.admit_us;
                if (result.entry.is_pinned) {
                    diagnostic_stats_.pinned_admit_cuda_pinned++;
                } else {
                    diagnostic_stats_.pinned_admit_fallback_ram++;
                }
                pinned_cache_[result.expert_key] = std::move(result.entry);
                LLAMA_DSTORAGE_DEBUG_LOG(
                        "DStorage DEBUG pinned:async_complete key=0x%" PRIx64 " slot=%d entries=%zu used_mib=%.2f\n",
                        result.expert_key, result.slot_idx, pinned_cache_.size(),
                        pinned_used_bytes_ / 1024.0 / 1024.0);
            } else {
                diagnostic_stats_.pinned_admit_dropped++;
                diagnostic_stats_.pinned_admit_us += result.admit_us;
                if (result.entry.host_ptr != nullptr) {
                    ds_loader_host_free(result.entry.host_ptr, result.entry.is_pinned ? 1 : 0);
                }
                if (pinned_used_bytes_ >= result.entry.bytes) {
                    pinned_used_bytes_ -= result.entry.bytes;
                } else {
                    pinned_used_bytes_ = 0;
                }
                LLAMA_DSTORAGE_DEBUG_LOG(
                        "DStorage DEBUG pinned:async_drop key=0x%" PRIx64 " slot=%d ok=%d used_mib=%.2f\n",
                        result.expert_key, result.slot_idx, result.ok ? 1 : 0,
                        pinned_used_bytes_ / 1024.0 / 1024.0);
            }
        }

        pinned_admissions_.erase(pinned_admissions_.begin() + i);
    }
#else
    GGML_UNUSED(wait_all);
#endif
}

void DStorageSlotManager::destroy_pinned_cache() {
#if LLAMA_DSTORAGE_HAS_BACKEND
    collect_pinned_admissions(true);
    for (auto & [key, entry] : pinned_cache_) {
        GGML_UNUSED(key);
        if (entry.host_ptr != nullptr) {
            ds_loader_host_free(entry.host_ptr, entry.is_pinned ? 1 : 0);
            entry.host_ptr = nullptr;
        }
    }
#endif
    pinned_cache_.clear();
    pinned_frequency_.clear();
    static_pinned_experts_.clear();
    static_pinned_expert_selection_.clear();
    static_pinned_preload_done_ = false;
    pinned_admissions_.clear();
    pinned_used_bytes_ = 0;
    pinned_pending_bytes_ = 0;
    pinned_tick_ = 0;
}

void DStorageSlotManager::release_prefill_only_pinned_cache() {
    if (!llama_dstorage_pinned_prefill_only_enabled() ||
            prefill_only_pinned_released_ ||
            pinned_cache_.empty()) {
        return;
    }

    const uint64_t min_prefill_pinned_bytes =
            llama_dstorage_pinned_prefill_only_min_pinned_bytes(pinned_budget_bytes_);
    if (prefill_stats_.stream_pinned_bytes < min_prefill_pinned_bytes) {
        return;
    }

    const int64_t t0 = llama_dstorage_now_us();
    const uint64_t released_entries = pinned_cache_.size();
    const uint64_t released_bytes = pinned_used_bytes_;

    destroy_pinned_cache();
    prefill_only_pinned_released_ = true;

    diagnostic_stats_.pinned_prefill_only_releases++;
    diagnostic_stats_.pinned_prefill_only_release_entries += released_entries;
    diagnostic_stats_.pinned_prefill_only_release_bytes += released_bytes;
    diagnostic_stats_.pinned_prefill_only_release_us +=
            uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - t0));

    LLAMA_LOG_INFO(
            "%s: released static pinned prefill cache entries=%" PRIu64
            " bytes=%.2f MiB release_us=%" PRIu64 "\n",
            __func__,
            released_entries,
            released_bytes / 1024.0 / 1024.0,
            uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - t0)));
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

bool DStorageSlotManager::init(
        int n_layers,
        int n_experts,
        int n_expert_used,
        uint32_t moe_gpu_cache_mib,
        uint32_t moe_pinned_cache_mib,
        bool phase_cache_policy) {
    LLAMA_DSTORAGE_DEBUG_LOG(
            "DStorage DEBUG slots:init enter n_layers=%d n_experts=%d n_expert_used=%d cache_mib=%u pinned_cache_mib=%u phase_cache_policy=%d\n",
            n_layers, n_experts, n_expert_used, moe_gpu_cache_mib, moe_pinned_cache_mib, phase_cache_policy ? 1 : 0);

    n_layers_      = n_layers;
    n_experts_     = n_experts;
    n_expert_used_ = n_expert_used;
    layer_reload_bytes_.assign(std::max(0, n_layers_), 0);

    uint32_t prefill_mib = calculate_phase_cache_mib(
            moe_gpu_cache_mib, phase_cache_policy, true);
    uint32_t decode_mib = calculate_phase_cache_mib(
            moe_gpu_cache_mib, phase_cache_policy, false);

    if (phase_cache_policy) {
        const char * env_prefill = std::getenv("LLAMA_DSTORAGE_PREFILL_CACHE_MIB");
        const char * env_decode = std::getenv("LLAMA_DSTORAGE_DECODE_CACHE_MIB");

        if (env_prefill && env_prefill[0] != '\0') {
            prefill_mib = std::max(256, std::atoi(env_prefill));
        }

        if (env_decode && env_decode[0] != '\0') {
            decode_mib = std::max(256, std::atoi(env_decode));
        }
    }

    pool_budget_bytes_prefill_ = uint64_t(prefill_mib) * 1024ull * 1024ull;
    pool_budget_bytes_decode_  = uint64_t(decode_mib) * 1024ull * 1024ull;

    // Start in prefill mode
    pool_budget_bytes_ = pool_budget_bytes_prefill_;
    current_allocated_phase_ = dstorage_moe_phase::prefill;

    pinned_budget_bytes_ = uint64_t(moe_pinned_cache_mib) * 1024ull * 1024ull;
    phase_cache_policy_ = phase_cache_policy;
    hybrid_arc_policy_ = llama_dstorage_hybrid_arc_requested();
    pinned_used_bytes_ = 0;
    pinned_pending_bytes_ = 0;
    pinned_tick_ = 0;
    static_pinned_layers_ = llama_dstorage_parse_static_pinned_layers(n_layers_);
    static_pinned_expert_selection_ =
            llama_dstorage_parse_static_pinned_experts(n_layers_, n_experts_);
    static_pinned_experts_.clear();
    static_pinned_preload_done_ = false;
    prefill_only_pinned_released_ = false;
    pool_group_slot_counts_.clear();
    layer_pool_groups_.clear();
    n_slots_ = 0;
    pools_allocated_ = false;
    use_tick_ = 0;
    prefill_stats_ = {};
    decode_stats_ = {};
    prefetch_stats_ = {};
    diagnostic_stats_ = {};
    request_activation_stats_ = {};
    last_request_phase_ = dstorage_moe_phase::prefill;
    request_tracking_started_ = false;
    execution_slots_by_layer_.clear();
    transfer_ewma_us_ = 0.0;
    layer_compute_ewma_us_ = 0.0;
    resident_experts_.clear();
    {
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        global_prefetch_queue_.clear();
    }
    reset_arc_state();
    routing_trace_sequence_ = 0;
    if (routing_trace_file_ != nullptr) {
        std::fclose(routing_trace_file_);
        routing_trace_file_ = nullptr;
    }

    const char * routing_trace_path = llama_dstorage_routing_trace_path();
    if (routing_trace_path != nullptr) {
        routing_trace_file_ = std::fopen(routing_trace_path, "w");
        if (routing_trace_file_ == nullptr) {
            LLAMA_LOG_WARN("%s: failed to open routing trace '%s'\n", __func__, routing_trace_path);
        } else {
            LLAMA_LOG_INFO("%s: recording MoE routing trace to %s\n", __func__, routing_trace_path);
        }
    }

#if LLAMA_DSTORAGE_HAS_BACKEND
    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:init before ds_loader_available\n");
    if (!ds_loader_available()) {
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:init ds_loader_available=false hr=0x%08" PRIx32 "\n",
                uint32_t(ds_loader_get_hresult()));
        LLAMA_LOG_WARN("%s: DirectStorage/GDS runtime is not available\n", __func__);
        return false;
    }
    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:init ds_loader_available=true\n");

    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:init before ds_loader_create\n");
    ds_loader_ = ds_loader_create();
    if (ds_loader_ == nullptr) {
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:init ds_loader_create=null hr=0x%08" PRIx32 "\n",
                uint32_t(ds_loader_get_hresult()));
        LLAMA_LOG_ERROR("%s: failed to create DirectStorage/GDS loader, hr=0x%08" PRIx32 "\n",
                __func__, uint32_t(ds_loader_get_hresult()));
        return false;
    }
    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:init ds_loader_create ok handle=%p\n", (void *) ds_loader_);

    if ((!static_pinned_layers_.empty() || !static_pinned_expert_selection_.empty()) &&
            pinned_budget_bytes_ == 0) {
        LLAMA_LOG_WARN("%s: static pinned experts were configured, but pinned host cache budget is 0 MiB\n", __func__);
    }

    LLAMA_LOG_INFO("%s: DirectStorage/GDS MoE initialized with %.2f MiB GPU cache budget, %.2f MiB pinned host cache budget, phase cache policy %s, cache policy %s, static pinned layers %zu, selected static experts %zu\n",
            __func__, pool_budget_bytes_ / 1024.0 / 1024.0, pinned_budget_bytes_ / 1024.0 / 1024.0,
            phase_cache_policy_ ? "enabled" : "disabled",
            hybrid_arc_policy_ ? "hybrid_arc" : "legacy",
            static_pinned_layers_.size(),
            static_pinned_expert_selection_.size());
    return true;
#else
    LLAMA_LOG_WARN("%s: DirectStorage/GDS MoE is only supported on Windows and Linux\n", __func__);
    return false;
#endif
}

bool DStorageSlotManager::ensure_pools_allocated() {
    const int64_t t_total0 = llama_dstorage_now_us();
    if (pools_allocated_) {
        llama_dstorage_trace_us("slots.ensure_pools", -1, "already_allocated", t_total0, llama_dstorage_now_us());
        return true;
    }

    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_pools enter type_count=%zu budget_mib=%.2f\n",
            type_strides_.size(), pool_budget_bytes_ / 1024.0 / 1024.0);

    int64_t t0 = llama_dstorage_now_us();
    
    auto align_stride_for_type = [&](const std::string & type_key, uint64_t stride) {
        uint64_t current_stride = stride;
        bool all_aligned = false;
        while (!all_aligned) {
            all_aligned = true;
            for (const auto & [tensor_key, info] : tensor_registry_) {
                std::string t_key = get_tensor_type_key(tensor_key.substr(tensor_key.find(':') + 1));
                if (t_key == type_key && info.type_size_row > 0) {
                    if (current_stride % info.type_size_row != 0) {
                        uint64_t rem = current_stride % info.type_size_row;
                        current_stride += (info.type_size_row - rem);
                        all_aligned = false;
                        break;
                    }
                }
            }
        }
        if (llama_dstorage_align_pool_stride_4k() && (current_stride % 4096) != 0) {
            current_stride += 4096 - (current_stride % 4096);
        }
        if (stride != current_stride) {
            LLAMA_LOG_INFO("%s: Rounding expert type '%s' stride from %" PRIu64 " to %" PRIu64 " bytes for block/optional 4K alignment\n",
                    __func__, type_key.c_str(), stride, current_stride);
        }
        return current_stride;
    };

    std::unordered_map<int, std::unordered_map<std::string, uint64_t>> group_strides;
    std::unordered_map<std::string, uint64_t> max_strides;
    std::unordered_map<int, uint64_t> layer_down_strides;
    for (const auto & [tensor_key, info] : tensor_registry_) {
        const size_t colon = tensor_key.find(':');
        const int layer_idx = colon == std::string::npos ? -1 : std::atoi(tensor_key.substr(0, colon).c_str());
        const std::string tensor_name = colon == std::string::npos ? tensor_key : tensor_key.substr(colon + 1);
        const std::string type_key = get_tensor_type_key(tensor_name);
        max_strides[type_key] = std::max(max_strides[type_key], info.expert_stride);
        if (type_key == "ffn_down_exps.weight") {
            layer_down_strides[layer_idx] = info.expert_stride;
        }
    }

    uint64_t min_down_stride = 0;
    uint64_t max_down_stride = 0;
    for (const auto & [layer, stride] : layer_down_strides) {
        GGML_UNUSED(layer);
        if (min_down_stride == 0 || stride < min_down_stride) {
            min_down_stride = stride;
        }
        max_down_stride = std::max(max_down_stride, stride);
    }

    const char * split_env = std::getenv("LLAMA_DSTORAGE_SPLIT_DOWN_POOLS");
    const bool split_down_pools =
            split_env == nullptr ||
            (std::strcmp(split_env, "0") != 0 &&
             std::strcmp(split_env, "false") != 0 &&
             std::strcmp(split_env, "FALSE") != 0);

    layer_pool_groups_.clear();
    int n_pool_groups = 1;
    if (split_down_pools && min_down_stride > 0 && max_down_stride > min_down_stride) {
        n_pool_groups = 2;
        for (const auto & [layer, stride] : layer_down_strides) {
            layer_pool_groups_[layer] = stride > min_down_stride ? 1 : 0;
        }
    }

    for (const auto & [type_key, stride] : max_strides) {
        if (n_pool_groups == 2 && type_key == "ffn_down_exps.weight") {
            group_strides[0][type_key] = align_stride_for_type(type_key, min_down_stride);
            group_strides[1][type_key] = align_stride_for_type(type_key, max_down_stride);
        } else {
            const uint64_t aligned = align_stride_for_type(type_key, stride);
            for (int group = 0; group < n_pool_groups; ++group) {
                group_strides[group][type_key] = aligned;
            }
        }
        type_strides_[type_key] = align_stride_for_type(type_key, stride);
    }

    std::vector<uint64_t> group_slot_bytes(n_pool_groups, 0);
    uint64_t bytes_per_logical_slot = 0;
    for (int group = 0; group < n_pool_groups; ++group) {
        for (const auto & [type_key, stride] : group_strides[group]) {
            group_slot_bytes[group] += stride;
        }
        bytes_per_logical_slot = std::max(bytes_per_logical_slot, group_slot_bytes[group]);
    }
    for (const auto & [type_key, stride] : type_strides_) {
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:stride type=%s stride_mib=%.4f\n",
                type_key.c_str(), stride / 1024.0 / 1024.0);
    }

    bool fused_expert_pools = llama_dstorage_fused_expert_pools();
    std::vector<uint64_t> fused_record_strides(n_pool_groups, 0);
    std::vector<std::unordered_map<std::string, uint64_t>> fused_tensor_offsets(n_pool_groups);
    if (fused_expert_pools) {
        for (const auto & [tensor_key, info] : tensor_registry_) {
            const size_t colon = tensor_key.find(':');
            const int layer_idx = colon == std::string::npos ? -1 : std::atoi(tensor_key.substr(0, colon).c_str());
            const std::string tensor_name = colon == std::string::npos ? tensor_key : tensor_key.substr(colon + 1);
            const std::string type_key = get_tensor_type_key(tensor_name);
            const int group = pool_group_for_layer(layer_idx);
            if (group < 0 || group >= n_pool_groups ||
                    !info.sidecar_expert_major ||
                    info.sidecar_record_stride == 0 ||
                    info.sidecar_tensor_offset + info.expert_stride > info.sidecar_record_stride) {
                fused_expert_pools = false;
                break;
            }
            if (fused_record_strides[group] == 0) {
                fused_record_strides[group] = info.sidecar_record_stride;
            } else if (fused_record_strides[group] != info.sidecar_record_stride) {
                fused_expert_pools = false;
                break;
            }
            auto offset_it = fused_tensor_offsets[group].find(type_key);
            if (offset_it == fused_tensor_offsets[group].end()) {
                fused_tensor_offsets[group][type_key] = info.sidecar_tensor_offset;
            } else if (offset_it->second != info.sidecar_tensor_offset) {
                fused_expert_pools = false;
                break;
            }
        }
        for (int group = 0; fused_expert_pools && group < n_pool_groups; ++group) {
            if (fused_record_strides[group] == 0 ||
                    fused_tensor_offsets[group].size() != group_strides[group].size()) {
                fused_expert_pools = false;
            }
        }
        if (fused_expert_pools) {
            for (int group = 0; group < n_pool_groups; ++group) {
                group_slot_bytes[group] = fused_record_strides[group];
            }
            bytes_per_logical_slot = 0;
            for (uint64_t bytes : group_slot_bytes) {
                bytes_per_logical_slot = std::max(bytes_per_logical_slot, bytes);
            }
            LLAMA_LOG_INFO("%s: using fused expert-major CUDA pools for direct whole-record GDS reads\n", __func__);
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_DSTORAGE_FUSED_EXPERT_POOLS requested, but expert-major layout validation failed; using separate tensor pools\n", __func__);
        }
    }
    llama_dstorage_trace_us("slots.ensure_pools", -1, "sum_type_strides", t0, llama_dstorage_now_us());

    const CacheLayout cache_layout = calculate_cache_layout(
            pool_budget_bytes_,
            bytes_per_logical_slot,
            n_layers_,
            n_experts_);
    const uint64_t persistent_cache_budget = cache_layout.persistent_cache_budget_bytes;
    const uint64_t active_staging_reserve = cache_layout.workspace_reserve_bytes;

    t0 = llama_dstorage_now_us();
    n_slots_ = cache_layout.n_persistent_slots;
    llama_dstorage_trace_us("slots.ensure_pools", -1, "calculate_slot_count", t0, llama_dstorage_now_us());
    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:calculated bytes_per_slot=%.4f MiB workspace_mib=%.2f persistent_budget_mib=%.2f n_slots=%d theoretical_cap=%" PRIu64 "\n",
            bytes_per_logical_slot / 1024.0 / 1024.0,
            active_staging_reserve / 1024.0 / 1024.0,
            persistent_cache_budget / 1024.0 / 1024.0,
            n_slots_,
            uint64_t(n_layers_) * uint64_t(n_experts_));
    if (n_slots_ <= 0) {
        LLAMA_LOG_ERROR("%s: DirectStorage MoE cache budget %.2f MiB cannot fit one logical expert slot (%.2f MiB)\n",
                __func__, persistent_cache_budget / 1024.0 / 1024.0,
                bytes_per_logical_slot / 1024.0 / 1024.0);
        return false;
    }

    pool_group_slot_counts_.assign(n_pool_groups, 0);
    if (n_pool_groups == 1) {
        pool_group_slot_counts_[0] = n_slots_;
    } else {
        const char * large_slots_env = std::getenv("LLAMA_DSTORAGE_LARGE_DOWN_SLOTS");
        int large_slots = large_slots_env != nullptr ? std::atoi(large_slots_env) : 0;
        if (large_slots <= 0) {
            const char * spec_env = std::getenv("LLAMA_DSTORAGE_SPECULATIVE_SLOTS");
            const int speculative_reserve = spec_env != nullptr && spec_env[0] != '\0'
                    ? calculate_speculative_slot_count(n_slots_, n_expert_used_)
                    : std::min(n_slots_ / 4, std::max(32, n_expert_used_ * 8));
            large_slots = std::max(
                    max_prefill_workspace_experts() + speculative_reserve,
                    (n_slots_ + 7) / 8);
        }
        large_slots = std::max(1, std::min(n_slots_ - 1, large_slots));
        pool_group_slot_counts_[1] = large_slots;
        pool_group_slot_counts_[0] = n_slots_ - large_slots;
    }

    t0 = llama_dstorage_now_us();
    slots_.assign(n_slots_, {});
    reset_arc_state();
    int global_slot = 0;
    for (int group = 0; group < n_pool_groups; ++group) {
        for (int local = 0; local < pool_group_slot_counts_[group]; ++local, ++global_slot) {
            slots_[global_slot].pool_group = group;
            slots_[global_slot].pool_index = local;
        }
    }
    for (int i = 0; i < n_slots_; i++) {
        slots_[i].layer_idx      = -1;
        slots_[i].expert_idx     = -1;
        slots_[i].occupied       = false;
        slots_[i].last_used_tick = 0;
    }
    llama_dstorage_trace_us("slots.ensure_pools", -1, "init_slot_table", t0, llama_dstorage_now_us());

    LLAMA_LOG_INFO("%s: DirectStorage MoE cache: %.2f MiB requested, %.2f MiB effective, %.2f MiB max logical slot, %d global slots, %d pool groups\n",
            __func__, pool_budget_bytes_ / 1024.0 / 1024.0,
            persistent_cache_budget / 1024.0 / 1024.0,
            bytes_per_logical_slot / 1024.0 / 1024.0, n_slots_, n_pool_groups);
    LLAMA_LOG_INFO("%s: DirectStorage MoE active workspace: %.2f MiB, persistent cache allocation budget: %.2f MiB\n",
            __func__, active_staging_reserve / 1024.0 / 1024.0,
            persistent_cache_budget / 1024.0 / 1024.0);

    uint64_t allocated_pool_bytes = 0;
    for (int group = 0; group < n_pool_groups; ++group) {
        LLAMA_LOG_INFO("%s: pool group %d has %d slots, %.2f MiB logical slot\n",
                __func__, group, pool_group_slot_counts_[group],
                group_slot_bytes[group] / 1024.0 / 1024.0);
        if (fused_expert_pools) {
            t0 = llama_dstorage_now_us();
            const uint64_t record_stride = fused_record_strides[group];
            const uint64_t pool_size = uint64_t(pool_group_slot_counts_[group]) * record_stride;
            allocated_pool_bytes += pool_size;

#if LLAMA_DSTORAGE_HAS_BACKEND
            const uint64_t alloc_size = llama_dstorage_pool_alloc_size(pool_size);
            const uint64_t alloc_ptr = ds_loader_cuda_alloc(alloc_size);
            if (alloc_ptr == 0) {
                LLAMA_LOG_ERROR("%s: failed to allocate fused CUDA VRAM pool for group %d (%.2f MiB)\n",
                        __func__, group, pool_size / 1024.0 / 1024.0);
                return false;
            }
            const uint64_t pool_ptr = llama_dstorage_pool_visible_ptr(alloc_ptr);
            bool first_view = true;
            for (const auto & [type_key, offset] : fused_tensor_offsets[group]) {
                const std::string pool_key = make_pool_key(group, type_key);
                type_pools_[pool_key] = {
                        pool_ptr + offset,
                        first_view ? alloc_ptr : 0,
                        record_stride,
                        first_view };
                first_view = false;
                LLAMA_LOG_INFO("%s: fused VRAM pool view group %d '%s': base=0x%" PRIx64 " offset=%.2f MiB stride=%.2f MiB owns=%d\n",
                        __func__, group, type_key.c_str(),
                        pool_ptr + offset,
                        offset / 1024.0 / 1024.0,
                        record_stride / 1024.0 / 1024.0,
                        type_pools_[pool_key].owns_alloc ? 1 : 0);
            }
#else
            bool first_view = true;
            for (const auto & [type_key, offset] : fused_tensor_offsets[group]) {
                const std::string pool_key = make_pool_key(group, type_key);
                GGML_UNUSED(offset);
                type_pools_[pool_key] = {0, 0, record_stride, first_view};
                first_view = false;
            }
#endif
            llama_dstorage_trace_us("slots.ensure_pools", -1, "fused_pool", t0, llama_dstorage_now_us());
            continue;
        }
        for (const auto & [type_key, stride] : group_strides[group]) {
            t0 = llama_dstorage_now_us();
            const uint64_t pool_size = uint64_t(pool_group_slot_counts_[group]) * stride;
            allocated_pool_bytes += pool_size;
            const std::string pool_key = make_pool_key(group, type_key);

            LLAMA_LOG_INFO("%s: allocating VRAM pool for group %d type '%s': %d slots x %.2f MiB = %.2f MiB total\n",
                    __func__, group, type_key.c_str(), pool_group_slot_counts_[group],
                    stride / 1024.0 / 1024.0,
                    pool_size / 1024.0 / 1024.0);

#if LLAMA_DSTORAGE_HAS_BACKEND
            LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:before cuda alloc group=%d type=%s pool_mib=%.2f\n",
                    group, type_key.c_str(), pool_size / 1024.0 / 1024.0);
            const uint64_t alloc_size = llama_dstorage_pool_alloc_size(pool_size);
            const uint64_t alloc_ptr = ds_loader_cuda_alloc(alloc_size);
            if (alloc_ptr == 0) {
                LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:cuda alloc failed group=%d type=%s pool_mib=%.2f\n",
                        group, type_key.c_str(), pool_size / 1024.0 / 1024.0);
                LLAMA_LOG_ERROR("%s: failed to allocate CUDA VRAM pool for group %d type '%s' (%.2f MiB)\n",
                        __func__, group, type_key.c_str(), pool_size / 1024.0 / 1024.0);
                return false;
            }
            const uint64_t pool_ptr = llama_dstorage_pool_visible_ptr(alloc_ptr);
            LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:cuda alloc ok group=%d type=%s ptr=0x%" PRIx64 "\n",
                    group, type_key.c_str(), pool_ptr);
            LLAMA_LOG_INFO("%s: VRAM pool for group %d '%s' allocated at 0x%" PRIx64 " visible=0x%" PRIx64 " align4k=%d\n",
                    __func__, group, type_key.c_str(), alloc_ptr, pool_ptr,
                    llama_dstorage_align_pool_base_4k() ? 1 : 0);
            type_pools_[pool_key] = {pool_ptr, alloc_ptr, stride};
#else
            type_pools_[pool_key] = {0, 0, stride};
#endif
            llama_dstorage_trace_us("slots.ensure_pools", -1, pool_key.c_str(), t0, llama_dstorage_now_us());
        }
    }
    LLAMA_LOG_INFO("%s: DirectStorage compacted pool allocation %.2f MiB, saved %.2f MiB vs max-stride layout\n",
            __func__,
            allocated_pool_bytes / 1024.0 / 1024.0,
            (uint64_t(n_slots_) * bytes_per_logical_slot - allocated_pool_bytes) / 1024.0 / 1024.0);

    pools_allocated_ = true;
    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_pools complete n_slots=%d type_pools=%zu\n",
            n_slots_, type_pools_.size());
    llama_dstorage_trace_us("slots.ensure_pools", -1, "total", t_total0, llama_dstorage_now_us());
    return true;
}

bool DStorageSlotManager::ensure_active_pools_allocated(int active_capacity) {
    if (active_capacity <= 0) {
        return false;
    }
    if (active_capacity_ >= active_capacity && active_type_pools_.size() == type_strides_.size()) {
        return true;
    }

#if LLAMA_DSTORAGE_HAS_BACKEND
    for (auto & [key, pool] : active_type_pools_) {
        if (pool.cuda_ptr != 0 && pool.owns_alloc) {
            ds_loader_cuda_free(pool.alloc_ptr != 0 ? pool.alloc_ptr : pool.cuda_ptr);
            pool.cuda_ptr = 0;
            pool.alloc_ptr = 0;
        }
    }
#endif
    active_type_pools_.clear();
    active_capacity_ = 0;
    active_slot_count_ = 0;

    for (const auto & [type_key, stride] : type_strides_) {
        const uint64_t pool_size = uint64_t(active_capacity) * stride;
#if LLAMA_DSTORAGE_HAS_BACKEND
        const uint64_t alloc_size = llama_dstorage_pool_alloc_size(pool_size);
        const uint64_t alloc_ptr = ds_loader_cuda_alloc(alloc_size);
        if (alloc_ptr == 0) {
            LLAMA_LOG_ERROR("%s: failed to allocate active CUDA staging pool for type '%s' (%.2f MiB)\n",
                    __func__, type_key.c_str(), pool_size / 1024.0 / 1024.0);
            return false;
        }
        const uint64_t pool_ptr = llama_dstorage_pool_visible_ptr(alloc_ptr);
        active_type_pools_[type_key] = { pool_ptr, alloc_ptr, stride };
#else
        active_type_pools_[type_key] = { 0, 0, stride };
#endif
        LLAMA_DSTORAGE_DEBUG_LOG("%s: active staging pool type=%s capacity=%d stride_mib=%.4f ptr=0x%" PRIx64 "\n",
                __func__, type_key.c_str(), active_capacity, stride / 1024.0 / 1024.0,
                active_type_pools_[type_key].cuda_ptr);
    }

    active_capacity_ = active_capacity;
    return true;
}

#if 1
bool DStorageSlotManager::ensure_prefill_workspaces_allocated() {
    if (prefill_workspaces_allocated_) {
        return true;
    }
    if (!ensure_pools_allocated()) {
        return false;
    }
    if (n_experts_ <= 0 || type_strides_.empty()) {
        return false;
    }

    const int workspace_count = 2;
    prefill_workspaces_.clear();
    prefill_workspaces_.resize(workspace_count);

    uint64_t bytes_per_workspace = 0;
    for (const auto & [type_key, stride] : type_strides_) {
        GGML_UNUSED(type_key);
        bytes_per_workspace += uint64_t(n_experts_) * stride;
    }

    for (int workspace_idx = 0; workspace_idx < workspace_count; ++workspace_idx) {
        PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
        workspace.layer_idx = -1;
        workspace.ready = false;
        workspace.loading = false;
        workspace.failed = false;
        workspace.pools.clear();

        for (const auto & [type_key, stride] : type_strides_) {
            const uint64_t pool_size = uint64_t(n_experts_) * stride;
#if LLAMA_DSTORAGE_HAS_BACKEND
            const uint64_t alloc_size = llama_dstorage_pool_alloc_size(pool_size);
            const uint64_t alloc_ptr = ds_loader_cuda_alloc(alloc_size);
            if (alloc_ptr == 0) {
                LLAMA_LOG_ERROR(
                        "%s: failed to allocate prefill workspace %d type '%s' (%.2f MiB)\n",
                        __func__, workspace_idx, type_key.c_str(), pool_size / 1024.0 / 1024.0);
                return false;
            }
            const uint64_t pool_ptr = llama_dstorage_pool_visible_ptr(alloc_ptr);
            workspace.pools[type_key] = { pool_ptr, alloc_ptr, stride };
#else
            workspace.pools[type_key] = { 0, 0, stride };
#endif
        }
        LLAMA_LOG_INFO(
                "%s: allocated prefill workspace %d for one MoE layer batch: %.2f MiB (%d expert slots)\n",
                __func__, workspace_idx, bytes_per_workspace / 1024.0 / 1024.0, n_experts_);
    }

    prefill_workspaces_allocated_ = true;
    return true;
}

bool DStorageSlotManager::load_prefill_workspace_layer(
        int workspace_idx,
        int layer_idx,
        const std::vector<int32_t> & expert_ids) {
    if (workspace_idx < 0 || workspace_idx >= int(prefill_workspaces_.size()) ||
            layer_idx < 0 || layer_idx >= n_layers_) {
        return false;
    }

    // If no experts specified (async preload path), nothing to load — activation will load on demand
    if (expert_ids.empty()) {
        return true;
    }

#if !LLAMA_DSTORAGE_HAS_BACKEND
    GGML_UNUSED(workspace_idx);
    GGML_UNUSED(layer_idx);
    GGML_UNUSED(expert_ids);
    return false;
#else
    const int64_t t0 = llama_dstorage_now_us();
    std::vector<DSLoaderStreamRequest> batch_requests;
    uint64_t possible_run_count = 0;
    uint64_t possible_run_expert_count = 0;

    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        auto layer_it = layer_tensors_.find(layer_idx);
        if (layer_it == layer_tensors_.end()) {
            return false;
        }

        PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
        batch_requests.reserve(layer_it->second.size() * expert_ids.size());
        for (const std::string & tname : layer_it->second) {
            const std::string tkey = make_tensor_key(layer_idx, tname.c_str());
            auto reg_it = tensor_registry_.find(tkey);
            if (reg_it == tensor_registry_.end()) {
                return false;
            }
            const ExpertTensorInfo & info = reg_it->second;
            if (info.total_size_compressed > 0 || info.expert_stride == 0) {
                return false;
            }

            const std::string type_key = get_tensor_type_key(tname);
            auto pool_it = workspace.pools.find(type_key);
            if (pool_it == workspace.pools.end()) {
                return false;
            }
            const TensorTypePool & pool = pool_it->second;

            // Load each requested expert individually into the workspace slot
            for (size_t i = 0; i < expert_ids.size(); ++i) {
                const int eid = expert_ids[i];
                if (eid < 0 || eid >= n_experts_) {
                    continue;
                }
                DSLoaderStreamRequest req;
                req.file_path = info.file_path.c_str();
                req.file_offset = llama_dstorage_expert_file_offset(info, eid);
                req.size = info.expert_stride;
                req.cuda_dest_ptr = pool.cuda_ptr + uint64_t(i) * pool.slot_size;
                req.uncompressed_size = info.expert_stride;
                batch_requests.push_back(req);
            }
        }
        possible_run_count = uint64_t(layer_it->second.size());
        possible_run_expert_count = uint64_t(layer_it->second.size()) * uint64_t(expert_ids.size());
    }

    if (llama_dstorage_sort_batch_by_offset()) {
        std::stable_sort(
                batch_requests.begin(),
                batch_requests.end(),
                [](const DSLoaderStreamRequest & a, const DSLoaderStreamRequest & b) {
                    return a.file_offset < b.file_offset;
                });
    }

    uint64_t stream_file_bytes = 0;
    uint64_t stream_cuda_bytes = 0;
    for (const DSLoaderStreamRequest & req : batch_requests) {
        stream_file_bytes += req.size;
        stream_cuda_bytes += req.uncompressed_size;
    }

    const uint64_t max_staging_bytes = llama_dstorage_max_batch_staging_bytes();
    int result = 0;
    size_t chunk_begin = 0;
    uint64_t stream_chunks = 0;
    uint64_t stream_staging_bytes = 0;
    uint64_t stream_submit_us = 0;
    uint64_t stream_wait_us = 0;
    {
        std::lock_guard<std::mutex> stream_lock(prefill_workspace_stream_mutex_);
        while (chunk_begin < batch_requests.size()) {
            size_t chunk_end = chunk_begin;
            uint64_t chunk_staging_bytes = 0;
            while (chunk_end < batch_requests.size()) {
                const DSLoaderStreamRequest & req = batch_requests[chunk_end];
                const uint64_t align_offset = req.file_offset % 4096;
                const uint64_t request_staging_bytes =
                        ((chunk_staging_bytes + 4095) & ~4095ULL) + req.size + align_offset;
                if (chunk_end > chunk_begin && request_staging_bytes > max_staging_bytes) {
                    break;
                }
                chunk_staging_bytes = request_staging_bytes;
                ++chunk_end;
            }

            const int64_t submit_t0 = llama_dstorage_now_us();
            result = ds_loader_stream_to_cuda_batch(
                    ds_loader_,
                    batch_requests.data() + chunk_begin,
                    int(chunk_end - chunk_begin));
            const int64_t submit_t1 = llama_dstorage_now_us();
            const int64_t wait_t0 = llama_dstorage_now_us();
            const int wait_result = result == 0 ? ds_loader_cuda_wait_event(ds_loader_) : -1;
            const int64_t wait_t1 = llama_dstorage_now_us();
            stream_staging_bytes += chunk_staging_bytes;
            stream_chunks++;
            if (result != 0 || wait_result != 0) {
                result = -1;
                break;
            }
            chunk_begin = chunk_end;
        }
    }

    const uint64_t transfer_us = uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - t0));
    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        prefetch_stats_.calls++;
        if (result == 0) {
            prefetch_stats_.misses += uint64_t(expert_ids.size());
            prefetch_stats_.miss_calls++;
            prefetch_stats_.stream_calls++;
            prefetch_stats_.stream_cold_experts += uint64_t(expert_ids.size());
            prefetch_stats_.stream_possible_runs += possible_run_count;
            prefetch_stats_.stream_possible_run_experts += possible_run_expert_count;
            prefetch_stats_.stream_runs += possible_run_count;
            prefetch_stats_.stream_run_experts += possible_run_expert_count;
            prefetch_stats_.stream_requests += uint64_t(batch_requests.size());
            prefetch_stats_.stream_chunks += stream_chunks;
            prefetch_stats_.stream_file_bytes += stream_file_bytes;
            prefetch_stats_.stream_cuda_bytes += stream_cuda_bytes;
            prefetch_stats_.stream_staging_bytes += stream_staging_bytes;
            prefetch_stats_.stream_submit_us += stream_submit_us;
            prefetch_stats_.stream_wait_us += stream_wait_us;
            prefetch_stats_.stream_total_us += transfer_us;
            prefetch_stats_.stream_max_requests_per_call = std::max(
                    prefetch_stats_.stream_max_requests_per_call,
                    uint64_t(batch_requests.size()));
            prefetch_stats_.stream_max_chunks_per_call = std::max(
                    prefetch_stats_.stream_max_chunks_per_call,
                    stream_chunks);
        }
    }

    if (result != 0) {
        LLAMA_LOG_ERROR(
                "%s: failed to load prefill workspace %d for layer %d, hr=0x%08" PRIx32 "\n",
                __func__, workspace_idx, layer_idx, uint32_t(ds_loader_get_hresult()));
        return false;
    }

    LLAMA_DSTORAGE_DEBUG_LOG(
            "DStorage DEBUG prefill_workspace:loaded workspace=%d layer=%d requests=%zu chunks=%" PRIu64 " file_mib=%.2f transfer_us=%" PRIu64 "\n",
            workspace_idx,
            layer_idx,
            batch_requests.size(),
            stream_chunks,
            stream_file_bytes / 1024.0 / 1024.0,
            transfer_us);
    return true;
#endif
}

bool DStorageSlotManager::prefill_workspace_activate_layer(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected,
        std::unordered_map<std::string, uint64_t> & out_pool_ptrs,
        std::vector<int32_t> & out_id_map) {
    const int64_t t0 = llama_dstorage_now_us();
    LLAMA_DSTORAGE_DEBUG_LOG(
            "DStorage DEBUG prefill_workspace:activate enter layer=%d n_selected=%d\n",
            layer_idx, n_selected);
    if (expert_ids == nullptr || n_selected <= 0 || layer_idx < 0 || layer_idx >= n_layers_) {
        return false;
    }
    if (!ensure_prefill_workspaces_allocated()) {
        return false;
    }

    const int workspace_idx = layer_idx % int(prefill_workspaces_.size());
    std::future<bool> pending;
    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
        if (workspace.loading && workspace.layer_idx == layer_idx && workspace.future.valid()) {
            pending = std::move(workspace.future);
        }
    }

    if (pending.valid()) {
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG prefill_workspace:activate waiting workspace=%d layer=%d\n",
                workspace_idx, layer_idx);
        const bool ok = pending.get();
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
        workspace.loading = false;
        workspace.ready = ok;
        workspace.failed = !ok;
        if (!ok) {
            return false;
        }
    }

    bool need_sync_load = false;
    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        const PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
        need_sync_load = !(workspace.ready && workspace.layer_idx == layer_idx);
    }

    if (need_sync_load) {
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG prefill_workspace:activate sync_load workspace=%d layer=%d\n",
                workspace_idx, layer_idx);
        {
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
            if (workspace.loading && workspace.future.valid()) {
                pending = std::move(workspace.future);
            }
        }
        if (pending.valid()) {
            pending.get();
        }
        {
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
            workspace.layer_idx = layer_idx;
            workspace.ready = false;
            workspace.loading = false;
            workspace.failed = false;
        }
        const bool ok = load_prefill_workspace_layer(workspace_idx, layer_idx, std::vector<int32_t>(expert_ids, expert_ids + n_selected));
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
        workspace.ready = ok;
        workspace.failed = !ok;
        if (!ok) {
            return false;
        }
    }

    out_pool_ptrs.clear();
    out_id_map.resize(n_selected);
    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
        if (!workspace.ready || workspace.layer_idx != layer_idx) {
            return false;
        }

        auto layer_it = layer_tensors_.find(layer_idx);
        if (layer_it == layer_tensors_.end()) {
            return false;
        }
        for (const std::string & tname : layer_it->second) {
            const std::string type_key = get_tensor_type_key(tname);
            auto pool_it = workspace.pools.find(type_key);
            if (pool_it == workspace.pools.end()) {
                return false;
            }
            out_pool_ptrs[tname] = pool_it->second.cuda_ptr;
        }

        for (int i = 0; i < n_selected; ++i) {
            if (expert_ids[i] < 0 || expert_ids[i] >= n_experts_) {
                return false;
            }
            out_id_map[i] = expert_ids[i];
        }

        active_prefill_workspace_idx_ = workspace_idx;
        active_prefill_workspace_layer_ = layer_idx;

        std::vector<int32_t> unique_experts;
        unique_experts.reserve(n_selected);
        for (int i = 0; i < n_selected; ++i) {
            if (std::find(unique_experts.begin(), unique_experts.end(), expert_ids[i]) == unique_experts.end()) {
                unique_experts.push_back(expert_ids[i]);
            }
        }
        prefill_stats_.calls++;
        prefill_stats_.hits += uint64_t(unique_experts.size());
        prefill_stats_.all_hit_calls++;
        timeline_stats_.real_calls++;
        timeline_stats_.prefill_calls++;
        timeline_stats_.all_hit_calls++;
        timeline_stats_.selected_experts += uint64_t(n_selected);
        timeline_stats_.unique_experts += uint64_t(unique_experts.size());
        timeline_stats_.hit_experts += uint64_t(unique_experts.size());
        const uint64_t ensure_us = uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - t0));
        timeline_stats_.ensure_total_us += ensure_us;
    }

    LLAMA_DSTORAGE_DEBUG_LOG(
            "DStorage DEBUG prefill_workspace:activate ready workspace=%d layer=%d\n",
            workspace_idx, layer_idx);
    return true;
}

void DStorageSlotManager::prefill_workspace_preload_layers_async(
        int first_layer_idx,
        int n_layers) {
    if (n_layers <= 0 || first_layer_idx >= n_layers_) {
        return;
    }
    if (!ensure_prefill_workspaces_allocated()) {
        return;
    }

    for (int offset = 0; offset < n_layers; ++offset) {
        const int layer_idx = first_layer_idx + offset;
        if (layer_idx < 0 || layer_idx >= n_layers_) {
            break;
        }
        const int workspace_idx = layer_idx % int(prefill_workspaces_.size());

        bool should_submit = false;
        {
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
            if (workspace.layer_idx == layer_idx && (workspace.ready || workspace.loading)) {
                continue;
            }
            if (workspace.loading) {
                continue;
            }
            if (workspace_idx == active_prefill_workspace_idx_) {
                continue;
            }
            workspace.layer_idx = layer_idx;
            workspace.ready = false;
            workspace.loading = true;
            workspace.failed = false;
            should_submit = true;
        }

        if (!should_submit) {
            continue;
        }

        try {
            std::future<bool> future = std::async(
                    std::launch::async,
                    [this, workspace_idx, layer_idx]() {
                        try {
                            // Async preload doesn't know which experts will be needed;
                            // load will be done on-demand by prefill_workspace_activate_layer
                            return this->load_prefill_workspace_layer(workspace_idx, layer_idx, {});
                        } catch (...) {
                            return false;
                        }
                    });
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
            workspace.future = std::move(future);
        } catch (...) {
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            PrefillWorkspace & workspace = prefill_workspaces_[workspace_idx];
            workspace.loading = false;
            workspace.failed = true;
        }
    }
}
#endif

bool DStorageSlotManager::is_better_request_eviction_candidate(
        const ExpertSlot & candidate,
        const ExpertSlot & current_best,
        int current_layer) const {
    const auto score = [&](const ExpertSlot & slot) {
        const uint32_t request_decode_count = request_activation_stats_.count(
                slot.layer_idx,
                slot.expert_idx,
                dstorage_moe_phase::decode);
        const uint32_t request_prefill_count = request_activation_stats_.count(
                slot.layer_idx,
                slot.expert_idx,
                dstorage_moe_phase::prefill);
        if (!hybrid_arc_enabled() ||
                slot.layer_idx < 0 ||
                slot.layer_idx >= int(layer_arc_.size())) {
            int64_t score = calculate_retention_score(
                    slot,
                    request_decode_count,
                    request_prefill_count,
                    current_layer,
                    n_layers_);
            if (decode_hot_protected(slot)) {
                score += 100000000000ll + int64_t(decode_hot_score(
                            make_expert_key(slot.layer_idx, slot.expert_idx)));
            }
            return score;
        }

        const LayerArcState & arc = layer_arc_[slot.layer_idx];
        int64_t score = calculate_hybrid_retention_score(
                slot,
                request_decode_count,
                request_prefill_count,
                current_layer,
                n_layers_,
                arc.recent_residents + arc.frequent_residents,
                layer_target_slots(slot.layer_idx),
                arc.target_recent,
                arc.recent_residents,
                contiguous_resident_neighbors(slot),
                expert_reload_bytes(slot.layer_idx));
        if (decode_hot_protected(slot)) {
            score += 100000000000ll + int64_t(decode_hot_score(
                        make_expert_key(slot.layer_idx, slot.expert_idx)));
        }
        return score;
    };

    const int64_t candidate_score = score(candidate);
    const int64_t best_score = score(current_best);

    if (candidate_score != best_score) {
        return candidate_score < best_score;
    }
    return candidate.last_used_tick < current_best.last_used_tick;
}

// ---------------------------------------------------------------------------
// register_expert_tensor
// ---------------------------------------------------------------------------

void DStorageSlotManager::register_expert_tensor(
        int layer_idx,
        const char * tensor_name,
        const std::wstring & file_path,
        uint64_t file_offset,
        uint64_t total_size,
        int n_experts,
        uint64_t total_size_compressed,
        uint64_t block_size) {

    ExpertTensorInfo info;
    info.file_path     = file_path;
    info.file_offset   = file_offset;
    info.total_size    = total_size;
    info.n_experts     = n_experts;
    info.expert_stride = n_experts > 0 ? total_size / uint64_t(n_experts) : 0;
    info.total_size_compressed = total_size_compressed;
    info.type_size_row = block_size;

    if (total_size_compressed == 0) {
        const auto & sidecar_manifest = llama_dstorage_sidecar_manifest();
        const auto sidecar_it = sidecar_manifest.find(llama_dstorage_sidecar_key(layer_idx, tensor_name));
        if (sidecar_it != sidecar_manifest.end()) {
            const llama_dstorage_sidecar_entry & sidecar = sidecar_it->second;
            if (sidecar.total_size != total_size || sidecar.n_experts != n_experts) {
                LLAMA_LOG_ERROR(
                        "%s: sidecar entry for layer %d tensor '%s' mismatches GGUF metadata"
                        " (sidecar size=%" PRIu64 " experts=%d, GGUF size=%" PRIu64 " experts=%d); ignoring sidecar\n",
                        __func__, layer_idx, tensor_name,
                        sidecar.total_size, sidecar.n_experts,
                        total_size, n_experts);
            } else {
                info.file_path = std::wstring(sidecar.path.begin(), sidecar.path.end());
                info.file_offset = sidecar.offset;
                info.sidecar_expert_major = sidecar.expert_major;
                info.sidecar_record_stride = sidecar.record_stride;
                info.sidecar_tensor_offset = sidecar.tensor_offset;
                if (info.sidecar_expert_major &&
                        (info.sidecar_record_stride == 0 ||
                         info.sidecar_tensor_offset + info.expert_stride > info.sidecar_record_stride)) {
                    LLAMA_LOG_ERROR(
                            "%s: expert-major sidecar entry for layer %d tensor '%s' has invalid record layout"
                            " (record_stride=%" PRIu64 " tensor_offset=%" PRIu64 " expert_stride=%" PRIu64 "); ignoring sidecar\n",
                            __func__, layer_idx, tensor_name,
                            info.sidecar_record_stride, info.sidecar_tensor_offset, info.expert_stride);
                    info.file_path = file_path;
                    info.file_offset = file_offset;
                    info.sidecar_expert_major = false;
                    info.sidecar_record_stride = 0;
                    info.sidecar_tensor_offset = 0;
                }
                LLAMA_LOG_INFO(
                        "%s: using aligned sidecar for expert tensor '%s' layer=%d offset=%" PRIu64
                        " size=%.2f MiB layout=%s record_stride=%" PRIu64 " tensor_offset=%" PRIu64 " path=%s\n",
                        __func__, tensor_name, layer_idx, sidecar.offset,
                        total_size / 1024.0 / 1024.0,
                        info.sidecar_expert_major ? "expert_major" : "tensor_major",
                        info.sidecar_record_stride, info.sidecar_tensor_offset,
                        sidecar.path.c_str());
            }
        }
    }

    if (total_size_compressed > 0) {
        // Read compressed sizes table from file
#if defined(_WIN32)
        FILE * f = _wfopen(file_path.c_str(), L"rb");
#else
        const std::string narrow_path(file_path.begin(), file_path.end());
        FILE * f = std::fopen(narrow_path.c_str(), "rb");
#endif
        if (f) {
#if defined(_WIN32)
            _fseeki64(f, file_offset, SEEK_SET);
#else
            fseeko(f, off_t(file_offset), SEEK_SET);
#endif
            info.expert_sizes_comp.resize(n_experts);
            const size_t n_read = std::fread(&info.expert_sizes_comp[0], sizeof(uint32_t), n_experts, f);
            if (n_read != size_t(n_experts)) {
                LLAMA_LOG_ERROR("%s: failed to read compressed expert sizes table: got %zu of %d entries\n",
                        __func__, n_read, n_experts);
            }
            std::fclose(f);
        } else {
            LLAMA_LOG_ERROR("%s: failed to open file to read compressed expert sizes: %ls\n", __func__, file_path.c_str());
        }
    }

    LLAMA_DSTORAGE_DEBUG_LOG(
            "DStorage DEBUG slots:register layer=%d tensor=%s total_mib=%.2f n_experts=%d stride_mib=%.4f file_off=%" PRIu64 " total_comp_mib=%.2f\n",
            layer_idx, tensor_name, total_size / 1024.0 / 1024.0, n_experts,
            info.expert_stride / 1024.0 / 1024.0, file_offset, total_size_compressed / 1024.0 / 1024.0);

    const std::string tensor_key = make_tensor_key(layer_idx, tensor_name);
    const auto existing = tensor_registry_.find(tensor_key);
    if (existing != tensor_registry_.end() &&
            layer_idx >= 0 &&
            layer_idx < int(layer_reload_bytes_.size()) &&
            layer_reload_bytes_[layer_idx] >= existing->second.expert_stride) {
        layer_reload_bytes_[layer_idx] -= existing->second.expert_stride;
    }
    tensor_registry_[tensor_key] = info;
    if (layer_idx >= 0 && layer_idx < int(layer_reload_bytes_.size())) {
        layer_reload_bytes_[layer_idx] += info.expert_stride;
    }

    auto & tensors = layer_tensors_[layer_idx];
    if (std::find(tensors.begin(), tensors.end(), tensor_name) == tensors.end()) {
        tensors.emplace_back(tensor_name);
    }

    const std::string type_key = get_tensor_type_key(tensor_name);
    auto stride_it = type_strides_.find(type_key);
    if (stride_it != type_strides_.end() && stride_it->second != info.expert_stride) {
        LLAMA_LOG_WARN("%s: tensor type '%s' stride changed from %.2f MiB to %.2f MiB; using larger stride\n",
                __func__, type_key.c_str(),
                stride_it->second / 1024.0 / 1024.0,
                info.expert_stride / 1024.0 / 1024.0);
        stride_it->second = std::max(stride_it->second, info.expert_stride);
    } else if (info.expert_stride > 0) {
        type_strides_[type_key] = info.expert_stride;
    }
}

// ---------------------------------------------------------------------------
// ensure_experts_loaded
// ---------------------------------------------------------------------------

bool DStorageSlotManager::prefetch_experts(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected) {
    std::unordered_map<std::string, uint64_t> unused_pool_ptrs;
    std::vector<int32_t> unused_id_map;
    return ensure_experts_loaded(
            layer_idx,
            expert_ids,
            n_selected,
            unused_pool_ptrs,
            unused_id_map,
            dstorage_moe_phase::prefetch);
}

bool DStorageSlotManager::prewarm_vram_experts(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected) {
    if (expert_ids == nullptr || n_selected <= 0) {
        return true;
    }

    const int chunk_size = std::max(1, n_expert_used_);
    for (int offset = 0; offset < n_selected; offset += chunk_size) {
        const int count = std::min(chunk_size, n_selected - offset);
        std::unordered_map<std::string, uint64_t> unused_pool_ptrs;
        std::vector<int32_t> unused_id_map;
        if (!ensure_experts_loaded(
                    layer_idx,
                    expert_ids + offset,
                    count,
                    unused_pool_ptrs,
                    unused_id_map,
                    dstorage_moe_phase::decode)) {
            return false;
        }
        // Prewarm is not followed by graph execution, so its temporary
        // execution pins must be released immediately.
        release_layer_experts(layer_idx);
    }
    return true;
}

bool DStorageSlotManager::preload_static_pinned_layers() {
    if (static_pinned_preload_done_) {
        return true;
    }
    static_pinned_preload_done_ = true;

    if (static_pinned_layers_.empty() && static_pinned_expert_selection_.empty()) {
        return true;
    }
    if (!pinned_cache_enabled()) {
        LLAMA_LOG_WARN("%s: pinned cache is disabled; cannot preload static pinned experts\n", __func__);
        return true;
    }

#if !defined(__linux__)
    LLAMA_LOG_WARN("%s: static pinned expert preload is currently implemented for Linux/GGUF files\n", __func__);
    return true;
#else
    uint64_t loaded_entries = 0;
    uint64_t skipped_entries = 0;
    uint64_t failed_entries = 0;
    uint64_t loaded_bytes = 0;
    uint64_t pinned_allocs = 0;
    uint64_t fallback_allocs = 0;

    std::vector<std::pair<int, int>> preload_targets = static_pinned_expert_selection_;
    std::unordered_set<uint64_t> requested;
    requested.reserve(preload_targets.size() + static_pinned_layers_.size() * size_t(n_experts_));
    for (const auto & target : preload_targets) {
        requested.insert(make_expert_key(target.first, target.second));
    }
    for (int static_layer_idx : static_pinned_layers_) {
        for (int eid = 0; eid < n_experts_; ++eid) {
            const uint64_t expert_key = make_expert_key(static_layer_idx, eid);
            if (requested.insert(expert_key).second) {
                preload_targets.emplace_back(static_layer_idx, eid);
            }
        }
    }

    struct StaticRead {
        std::wstring file_path;
        uint64_t file_offset = 0;
        uint64_t dst_offset = 0;
        uint64_t size = 0;
    };
    struct PreloadJob {
        int layer_idx = -1;
        int eid = -1;
        uint64_t expert_key = 0;
        uint64_t entry_bytes = 0;
        std::vector<PinnedSlice> slices;
        std::vector<StaticRead> reads;
        void * host_ptr = nullptr;
        int is_pinned = 0;
        bool ok = false;
    };

    // Phase A: build the job plan serially (registry lookups + budget cutoff).
    // The actual file I/O is the bottleneck and is parallelized in phase B.
    std::vector<PreloadJob> jobs;
    jobs.reserve(preload_targets.size());
    uint64_t planned_bytes = pinned_used_bytes_;
    for (size_t target_idx = 0; target_idx < preload_targets.size(); ++target_idx) {
        const int static_layer_idx = preload_targets[target_idx].first;
        const int eid = preload_targets[target_idx].second;
        auto layer_it = layer_tensors_.find(static_layer_idx);
        if (layer_it == layer_tensors_.end()) {
            LLAMA_LOG_WARN("%s: no tensors registered for static pinned layer %d\n", __func__, static_layer_idx);
            skipped_entries++;
            continue;
        }
        const std::vector<std::string> & layer_tnames = layer_it->second;

        const uint64_t expert_key = make_expert_key(static_layer_idx, eid);
        if (pinned_cache_.find(expert_key) != pinned_cache_.end()) {
            static_pinned_experts_.insert(expert_key);
            continue;
        }

        PreloadJob job;
        job.layer_idx = static_layer_idx;
        job.eid = eid;
        job.expert_key = expert_key;
        job.slices.reserve(layer_tnames.size());
        job.reads.reserve(layer_tnames.size());

        bool can_preload = true;
        for (const std::string & tname : layer_tnames) {
            const std::string tkey = make_tensor_key(static_layer_idx, tname.c_str());
            auto reg_it = tensor_registry_.find(tkey);
            if (reg_it == tensor_registry_.end()) {
                can_preload = false;
                break;
            }
            const ExpertTensorInfo & info = reg_it->second;
            if (info.total_size_compressed > 0 || info.expert_stride == 0 || eid >= info.n_experts) {
                can_preload = false;
                break;
            }

            job.slices.push_back({ tname, job.entry_bytes, info.expert_stride });
            job.reads.push_back({
                    info.file_path,
                    llama_dstorage_expert_file_offset(info, eid),
                    job.entry_bytes,
                    info.expert_stride });
            job.entry_bytes += info.expert_stride;
        }

        if (!can_preload || job.entry_bytes == 0) {
            skipped_entries++;
            continue;
        }
        if (job.entry_bytes > pinned_budget_bytes_ || planned_bytes + job.entry_bytes > pinned_budget_bytes_) {
            skipped_entries += uint64_t(preload_targets.size() - target_idx);
            LLAMA_LOG_WARN(
                    "%s: static pinned budget exhausted after %zu entries (used %.2f MiB / budget %.2f MiB)\n",
                    __func__,
                    jobs.size(),
                    planned_bytes / 1024.0 / 1024.0,
                    pinned_budget_bytes_ / 1024.0 / 1024.0);
            break;
        }
        planned_bytes += job.entry_bytes;
        jobs.push_back(std::move(job));
    }

    // Phase B: allocate + read each job in parallel. The pinned host allocation
    // and the per-expert preads are the expensive parts; running them across a
    // worker pool turns the one-time preload from a serial SSD trickle into a
    // saturated multi-stream load. Each worker keeps its own file descriptors.
    if (!jobs.empty()) {
        const int hw = int(std::thread::hardware_concurrency());
        const int n_workers = std::max(1, std::min<int>(int(jobs.size()), hw > 0 ? hw : 8));
        std::atomic<size_t> next_job{0};
        auto worker = [&]() {
            std::unordered_map<std::string, int> fds;
            for (;;) {
                const size_t i = next_job.fetch_add(1, std::memory_order_relaxed);
                if (i >= jobs.size()) {
                    break;
                }
                PreloadJob & job = jobs[i];
                job.host_ptr = ds_loader_host_alloc(job.entry_bytes, &job.is_pinned);
                if (job.host_ptr == nullptr) {
                    continue;
                }
                bool ok = true;
                for (const StaticRead & read : job.reads) {
                    const std::string path(read.file_path.begin(), read.file_path.end());
                    int fd = -1;
                    auto it = fds.find(path);
                    if (it != fds.end()) {
                        fd = it->second;
                    } else {
                        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
                        fds.emplace(path, fd);
                    }
                    if (fd < 0) {
                        ok = false;
                        break;
                    }
                    char * dst = reinterpret_cast<char *>(job.host_ptr) + read.dst_offset;
                    if (!llama_dstorage_pread_full(fd, read.file_offset, dst, read.size)) {
                        ok = false;
                        break;
                    }
                }
                job.ok = ok;
            }
            for (auto & e : fds) {
                if (e.second >= 0) {
                    close(e.second);
                }
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(size_t(n_workers));
        for (int t = 0; t < n_workers; ++t) {
            workers.emplace_back(worker);
        }
        for (std::thread & th : workers) {
            th.join();
        }
    }

    // Phase C: commit successful jobs into the pinned cache serially.
    for (PreloadJob & job : jobs) {
        if (job.host_ptr == nullptr) {
            failed_entries++;
            continue;
        }
        if (!job.ok) {
            ds_loader_host_free(job.host_ptr, job.is_pinned);
            failed_entries++;
            continue;
        }

        PinnedEntry entry;
        entry.layer_idx = job.layer_idx;
        entry.expert_idx = job.eid;
        entry.host_ptr = job.host_ptr;
        entry.bytes = job.entry_bytes;
        entry.is_pinned = job.is_pinned != 0;
        entry.last_used_tick = ++pinned_tick_;
        entry.slices = std::move(job.slices);

        pinned_cache_[job.expert_key] = std::move(entry);
        pinned_used_bytes_ += job.entry_bytes;
        pinned_frequency_[job.expert_key] = std::max(
                pinned_frequency_[job.expert_key],
                std::numeric_limits<uint32_t>::max() / 2);
        static_pinned_experts_.insert(job.expert_key);
        loaded_entries++;
        loaded_bytes += job.entry_bytes;
        if (job.is_pinned != 0) {
            pinned_allocs++;
        } else {
            fallback_allocs++;
        }
    }

    LLAMA_LOG_INFO(
            "%s: static pinned preload loaded=%" PRIu64 " skipped=%" PRIu64
            " failed=%" PRIu64 " bytes=%.2f MiB entries=%zu used=%.2f/%.2f MiB"
            " cuda_pinned=%" PRIu64 " fallback_ram=%" PRIu64 "\n",
            __func__,
            loaded_entries,
            skipped_entries,
            failed_entries,
            loaded_bytes / 1024.0 / 1024.0,
            pinned_cache_.size(),
            pinned_used_bytes_ / 1024.0 / 1024.0,
            pinned_budget_bytes_ / 1024.0 / 1024.0,
            pinned_allocs,
            fallback_allocs);
    return true;
#endif
}

bool DStorageSlotManager::prewarm_pinned_experts(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected) {
    if (!pinned_cache_enabled()) {
        LLAMA_LOG_WARN("%s: pinned cache is disabled; cannot prewarm layer %d\n", __func__, layer_idx);
        return false;
    }
    if (expert_ids == nullptr || n_selected <= 0) {
        return true;
    }

    const int chunk_size = std::max(1, n_expert_used_);
    const uint32_t admit_min_hits = pinned_admit_min_hits();
    for (int offset = 0; offset < n_selected; offset += chunk_size) {
        const int count = std::min(chunk_size, n_selected - offset);
        {
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            for (int i = 0; i < count; ++i) {
                const int32_t eid = expert_ids[offset + i];
                if (eid >= 0) {
                    pinned_frequency_[make_expert_key(layer_idx, eid)] =
                        std::max(pinned_frequency_[make_expert_key(layer_idx, eid)], admit_min_hits);
                }
            }
        }
        if (!prewarm_vram_experts(layer_idx, expert_ids + offset, count)) {
            return false;
        }
    }
    collect_pinned_admissions(true);
    return true;
}

bool DStorageSlotManager::prefetch_global_batch(std::vector<QueuedGlobalPrefetch> batch) {
    if (batch.empty()) {
        return true;
    }

#if !LLAMA_DSTORAGE_HAS_BACKEND
    GGML_UNUSED(batch);
    return false;
#else
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    if (!ensure_pools_allocated()) {
        return false;
    }
    collect_pinned_admissions(false);

    const int speculative_slots = speculative_slot_count();
    if (speculative_slots <= 0) {
        return false;
    }

    struct GlobalPlan {
        int layer_idx = -1;
        int32_t eid = -1;
        int target_slot = -1;
    };

    std::vector<GlobalPlan> load_plan;
    std::vector<int> affected_layers;
    uint64_t possible_run_count = 0;
    uint64_t possible_run_expert_count = 0;

    auto slot_allowed = [&](int slot_idx, int layer_idx) {
        return slot_in_speculative_partition(slot_idx, speculative_slots) &&
               slot_compatible_with_layer(slot_idx, layer_idx);
    };
    auto slot_already_reserved = [&](int slot_idx) {
        for (const GlobalPlan & plan : load_plan) {
            if (plan.target_slot == slot_idx) {
                return true;
            }
        }
        return false;
    };
    auto choose_eviction_slot = [&](int current_layer) {
        int target_slot = -1;
        for (int s = 0; s < n_slots_; ++s) {
            if (!slot_allowed(s, current_layer) || slot_already_reserved(s) || slot_is_protected(slots_[s])) {
                continue;
            }
            if (target_slot < 0 ||
                    is_better_request_eviction_candidate(
                            slots_[s],
                            slots_[target_slot],
                            current_layer)) {
                target_slot = s;
            }
        }
        return target_slot;
    };

    for (QueuedGlobalPrefetch & request : batch) {
        if (request.layer_idx < 0 || request.layer_idx >= n_layers_) {
            continue;
        }
        if (std::find(affected_layers.begin(), affected_layers.end(), request.layer_idx) == affected_layers.end()) {
            affected_layers.push_back(request.layer_idx);
        }

        std::vector<int32_t> unique_experts;
        unique_experts.reserve(request.expert_ids.size());
        for (int32_t eid : request.expert_ids) {
            if (eid >= 0 && eid < n_experts_ &&
                    std::find(unique_experts.begin(), unique_experts.end(), eid) == unique_experts.end()) {
                unique_experts.push_back(eid);
            }
        }
        std::sort(unique_experts.begin(), unique_experts.end());

        prefetch_stats_.calls++;
        for (int32_t eid : unique_experts) {
            bool hit = false;
            for (ExpertSlot & slot : slots_) {
                if (slot.occupied && slot.layer_idx == request.layer_idx && slot.expert_idx == eid) {
                    slot.last_used_tick = ++use_tick_;
                    slot.prefetch_hits += 1000;
                    arc_record_hit(slot, dstorage_moe_phase::prefetch);
                    hit = true;
                    break;
                }
            }
            if (hit) {
                prefetch_stats_.hits++;
                continue;
            }

            int target_slot = -1;
            for (int s = 0; s < n_slots_; ++s) {
                if (slot_allowed(s, request.layer_idx) && !slots_[s].occupied && !slots_[s].admission_pending && !slot_already_reserved(s)) {
                    target_slot = s;
                    break;
                }
            }
            if (target_slot < 0) {
                target_slot = choose_eviction_slot(request.layer_idx);
            }
            if (target_slot < 0) {
                prefetch_stats_.misses++;
                continue;
            }

            if (slots_[target_slot].occupied) {
                if (decode_hot_protected(slots_[target_slot])) {
                    decode_hot_stats_.protected_evictions++;
                }
                if (slots_[target_slot].prefetch_hits > 0) {
                    if (slots_[target_slot].decode_hits == 0 && slots_[target_slot].prefill_hits == 0) {
                        diagnostic_stats_.prefetch_evicted_before_use++;
                    } else {
                        diagnostic_stats_.prefetch_evicted_after_use++;
                    }
                }
                arc_record_eviction(slots_[target_slot]);
                if (slots_[target_slot].decode_hits > 0) {
                    prefetch_stats_.evicted_decode_touched++;
                } else {
                    prefetch_stats_.evicted_prefill_only++;
                }
            }

            slots_[target_slot].layer_idx = -1;
            slots_[target_slot].expert_idx = -1;
            slots_[target_slot].occupied = false;
            slots_[target_slot].transfer_pending = true;
            slots_[target_slot].execution_pins = 0;
            slots_[target_slot].prefill_hits = 0;
            slots_[target_slot].decode_hits = 0;
            slots_[target_slot].prefetch_hits = 0;
            slots_[target_slot].arc_segment = dstorage_arc_segment::none;
            slots_[target_slot].contiguous_neighbors = 0;

            prefetch_stats_.misses++;
            load_plan.push_back({ request.layer_idx, eid, target_slot });
        }
        if (unique_experts.empty()) {
            prefetch_stats_.all_hit_calls++;
        }
    }

    if (load_plan.empty()) {
        return true;
    }

    std::vector<DSLoaderStreamRequest> batch_requests;
    batch_requests.reserve(load_plan.size() * 3);
    for (const GlobalPlan & plan : load_plan) {
        auto layer_it = layer_tensors_.find(plan.layer_idx);
        if (layer_it == layer_tensors_.end()) {
            return false;
        }

        for (const std::string & tname : layer_it->second) {
            const std::string tkey = make_tensor_key(plan.layer_idx, tname.c_str());
            auto reg_it = tensor_registry_.find(tkey);
            if (reg_it == tensor_registry_.end()) {
                return false;
            }
            const ExpertTensorInfo & info = reg_it->second;
            if (info.total_size_compressed > 0) {
                return false;
            }

            const std::string pool_key = pool_key_for_layer(plan.layer_idx, tname);
            auto pool_it = type_pools_.find(pool_key);
            if (pool_it == type_pools_.end()) {
                return false;
            }
            const TensorTypePool & pool = pool_it->second;
            const uint64_t file_off = llama_dstorage_expert_file_offset(info, plan.eid);
            const uint64_t dest = pool.cuda_ptr + uint64_t(slots_[plan.target_slot].pool_index) * pool.slot_size;
            DSLoaderStreamRequest req;
            req.file_path = info.file_path.c_str();
            req.file_offset = file_off;
            req.size = info.expert_stride;
            req.cuda_dest_ptr = dest;
            req.uncompressed_size = info.expert_stride;
            batch_requests.push_back(req);
        }
        possible_run_count++;
        possible_run_expert_count++;
    }

    if (llama_dstorage_sort_batch_by_offset()) {
        std::stable_sort(
                batch_requests.begin(),
                batch_requests.end(),
                [](const DSLoaderStreamRequest & a, const DSLoaderStreamRequest & b) {
                    return a.file_offset < b.file_offset;
                });
    }

    const int64_t t0 = llama_dstorage_now_us();
    uint64_t stream_file_bytes = 0;
    uint64_t stream_cuda_bytes = 0;
    for (const DSLoaderStreamRequest & req : batch_requests) {
        stream_file_bytes += req.size;
        stream_cuda_bytes += req.uncompressed_size;
    }

    const uint64_t max_staging_bytes = llama_dstorage_max_batch_staging_bytes();
    int result = 0;
    size_t chunk_begin = 0;
    uint64_t stream_chunks = 0;
    uint64_t stream_staging_bytes = 0;
    uint64_t stream_submit_us = 0;
    uint64_t stream_wait_us = 0;
    while (chunk_begin < batch_requests.size()) {
        size_t chunk_end = chunk_begin;
        uint64_t chunk_staging_bytes = 0;
        while (chunk_end < batch_requests.size()) {
            const DSLoaderStreamRequest & req = batch_requests[chunk_end];
            const uint64_t align_offset = req.file_offset % 4096;
            const uint64_t request_staging_bytes =
                    ((chunk_staging_bytes + 4095) & ~4095ULL) + req.size + align_offset;
            if (chunk_end > chunk_begin && request_staging_bytes > max_staging_bytes) {
                break;
            }
            chunk_staging_bytes = request_staging_bytes;
            ++chunk_end;
        }

        const int64_t submit_t0 = llama_dstorage_now_us();
        result = ds_loader_stream_to_cuda_batch(
                ds_loader_,
                batch_requests.data() + chunk_begin,
                int(chunk_end - chunk_begin));
        const int64_t submit_t1 = llama_dstorage_now_us();
        const int64_t wait_t0 = llama_dstorage_now_us();
        const int wait_result = result == 0 ? ds_loader_cuda_wait_event(ds_loader_) : -1;
        const int64_t wait_t1 = llama_dstorage_now_us();
        stream_submit_us += uint64_t(std::max<int64_t>(0, submit_t1 - submit_t0));
        stream_wait_us += uint64_t(std::max<int64_t>(0, wait_t1 - wait_t0));
        stream_staging_bytes += chunk_staging_bytes;
        stream_chunks++;
        if (result != 0 || wait_result != 0) {
            result = -1;
            break;
        }
        chunk_begin = chunk_end;
    }

    if (result != 0) {
        for (const GlobalPlan & plan : load_plan) {
            if (plan.target_slot >= 0 && plan.target_slot < int(slots_.size())) {
                slots_[plan.target_slot].transfer_pending = false;
            }
        }
        return false;
    }

    const uint64_t transfer_us = uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - t0));
    const double transfer_val = double(transfer_us);
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        transfer_ewma_us_ = transfer_ewma_us_ == 0.0
                ? transfer_val
                : transfer_ewma_us_ * 0.8 + transfer_val * 0.2;
    }
    prefetch_stats_.stream_calls++;
    prefetch_stats_.stream_cold_experts += uint64_t(load_plan.size());
    prefetch_stats_.stream_possible_runs += possible_run_count;
    prefetch_stats_.stream_possible_run_experts += possible_run_expert_count;
    prefetch_stats_.stream_runs += uint64_t(load_plan.size());
    prefetch_stats_.stream_run_experts += uint64_t(load_plan.size());
    prefetch_stats_.stream_requests += uint64_t(batch_requests.size());
    prefetch_stats_.stream_chunks += stream_chunks;
    prefetch_stats_.stream_file_bytes += stream_file_bytes;
    prefetch_stats_.stream_cuda_bytes += stream_cuda_bytes;
    prefetch_stats_.stream_staging_bytes += stream_staging_bytes;
    prefetch_stats_.stream_submit_us += stream_submit_us;
    prefetch_stats_.stream_wait_us += stream_wait_us;
    prefetch_stats_.stream_total_us += transfer_us;
    prefetch_stats_.stream_max_requests_per_call = std::max(
            prefetch_stats_.stream_max_requests_per_call,
            uint64_t(batch_requests.size()));
    prefetch_stats_.stream_max_chunks_per_call = std::max(
            prefetch_stats_.stream_max_chunks_per_call,
            stream_chunks);
    diagnostic_stats_.prefetch_loaded_experts += uint64_t(load_plan.size());

    for (const GlobalPlan & plan : load_plan) {
        ExpertSlot & slot = slots_[plan.target_slot];
        slot.layer_idx = plan.layer_idx;
        slot.expert_idx = plan.eid;
        slot.occupied = true;
        slot.transfer_pending = false;
        slot.last_used_tick = ++use_tick_;
        slot.prefetch_hits = 1000;
        arc_record_admission(slot, dstorage_moe_phase::prefetch);
    }

    LLAMA_DSTORAGE_DEBUG_LOG(
            "DStorage DEBUG prefetch:global_batch layers=%zu experts=%zu requests=%zu chunks=%" PRIu64 " file_mib=%.2f transfer_us=%" PRIu64 "\n",
            affected_layers.size(),
            load_plan.size(),
            batch_requests.size(),
            stream_chunks,
            stream_file_bytes / 1024.0 / 1024.0,
            transfer_us);
    return true;
#endif
}

bool DStorageSlotManager::prefetch_experts_async(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected) {
    if (expert_ids == nullptr || n_selected <= 0) {
        return false;
    }

    collect_async_prefetches(false);

    {
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        if (int(async_prefetches_.size()) >= llama_dstorage_force_prefetch_max_pending()) {
            return false;
        }
    }

    std::vector<int32_t> ids;
    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        diagnostic_stats_.prefetch_candidate_calls++;
        const bool high_confidence_prefetch = llama_dstorage_high_confidence_prefetch_enabled();
        const uint32_t observations = high_confidence_prefetch
                ? request_activation_stats_.rolling_decode_observations(layer_idx)
                : request_activation_stats_.decode_observations(layer_idx);
        const bool allow_route_prediction = llama_dstorage_predictive_prefetch_enabled();
        if ((high_confidence_prefetch || !allow_route_prediction) &&
                observations < uint32_t(llama_dstorage_prefetch_min_observations())) {
            diagnostic_stats_.prefetch_reject_observations += uint64_t(n_selected);
            return false;
        }

        bool has_free_slot = false;
        int64_t victim_retention_cost = std::numeric_limits<int64_t>::max();
        const int speculative_slots = speculative_slot_count();
        for (int slot_idx = 0; slot_idx < int(slots_.size()); ++slot_idx) {
            if (!slot_in_speculative_partition(slot_idx, speculative_slots)) {
                continue;
            }
            const ExpertSlot & slot = slots_[slot_idx];
            if (!slot.occupied) {
                has_free_slot = true;
                victim_retention_cost = 0;
                break;
            }
            if (slot_is_protected(slot)) {
                continue;
            }
            victim_retention_cost = std::min(
                    victim_retention_cost,
                    calculate_retention_score(
                            slot,
                            request_activation_stats_.count(
                                    slot.layer_idx,
                                    slot.expert_idx,
                                    dstorage_moe_phase::decode),
                            request_activation_stats_.count(
                                    slot.layer_idx,
                                    slot.expert_idx,
                                    dstorage_moe_phase::prefill),
                            layer_idx,
                            n_layers_));
        }
        diagnostic_stats_.prefetch_lowest_victim_cost = std::min(
                diagnostic_stats_.prefetch_lowest_victim_cost,
                victim_retention_cost);

        for (int i = 0; i < n_selected; ++i) {
            const int32_t eid = expert_ids[i];
            if (eid < 0 || std::find(ids.begin(), ids.end(), eid) != ids.end()) {
                continue;
            }

            bool already_resident = false;
            for (const ExpertSlot & slot : slots_) {
                if (slot.occupied && slot.layer_idx == layer_idx && slot.expert_idx == eid) {
                    already_resident = true;
                    break;
                }
            }
            if (already_resident) {
                diagnostic_stats_.prefetch_reject_resident++;
                continue;
            }

            const double min_confidence = llama_dstorage_prefetch_min_confidence();
            if (high_confidence_prefetch &&
                    observations < uint32_t(llama_dstorage_prefetch_min_observations())) {
                diagnostic_stats_.prefetch_reject_observations++;
                continue;
            }
            const double confidence = observations > 0
                    ? (high_confidence_prefetch
                            ? request_activation_stats_.rolling_decode_confidence(layer_idx, eid)
                            : request_activation_stats_.decode_confidence(layer_idx, eid))
                    : (high_confidence_prefetch ? 0.0 : min_confidence);
            diagnostic_stats_.prefetch_best_confidence =
                    std::max(diagnostic_stats_.prefetch_best_confidence, confidence);
            ExpertSlot candidate;
            candidate.layer_idx = layer_idx;
            candidate.expert_idx = eid;
            candidate.prefetch_hits = 1000;
            const uint32_t candidate_decode_count = high_confidence_prefetch
                    ? request_activation_stats_.rolling_decode_count(layer_idx, eid)
                    : request_activation_stats_.count(
                            layer_idx,
                            eid,
                            dstorage_moe_phase::decode);
            const int64_t candidate_value = calculate_retention_score(
                    candidate,
                    candidate_decode_count,
                    request_activation_stats_.count(
                            layer_idx,
                            eid,
                            dstorage_moe_phase::prefill),
                    layer_idx,
                    n_layers_);
            diagnostic_stats_.prefetch_best_candidate_value = std::max(
                    diagnostic_stats_.prefetch_best_candidate_value,
                    candidate_value);

            if (should_admit_prefetch(
                        confidence,
                        candidate_value,
                        victim_retention_cost,
                        has_free_slot,
                        min_confidence)) {
                ids.push_back(eid);
            } else {
                diagnostic_stats_.prefetch_reject_policy++;
            }
        }
    }

    if (ids.empty()) {
        return false;
    }

    const int admitted_count = int(ids.size());
    try {
        std::future<bool> future = std::async(
                std::launch::async,
                [this, layer_idx, ids = std::move(ids)]() mutable {
                    try {
                        return this->prefetch_experts(layer_idx, ids.data(), int(ids.size()));
                    } catch (...) {
                        return false;
                    }
                });
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        async_prefetches_.push_back({ layer_idx, admitted_count, { layer_idx }, std::move(future) });
        diagnostic_stats_.prefetch_submitted_calls++;
        diagnostic_stats_.prefetch_submitted_experts += uint64_t(admitted_count);
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG prefetch:async_start layer=%d ids=%d pending=%zu\n",
                layer_idx, admitted_count, async_prefetches_.size());
        return true;
    } catch (...) {
        return false;
    }
}

bool DStorageSlotManager::prefetch_experts_async_force(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected) {
    if (expert_ids == nullptr || n_selected <= 0) {
        return false;
    }

    collect_async_prefetches(false);

    std::vector<int32_t> ids;
    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        diagnostic_stats_.prefetch_candidate_calls++;
        const int speculative_slots = speculative_slot_count();
        if (speculative_slots <= 0) {
            diagnostic_stats_.prefetch_reject_policy += uint64_t(n_selected);
            return false;
        }

        ids.reserve(std::min(n_selected, speculative_slots));
        for (int i = 0; i < n_selected && int(ids.size()) < speculative_slots; ++i) {
            const int32_t eid = expert_ids[i];
            if (eid < 0 || eid >= n_experts_ ||
                    std::find(ids.begin(), ids.end(), eid) != ids.end()) {
                continue;
            }

            bool already_resident = false;
            for (const ExpertSlot & slot : slots_) {
                if (slot.occupied && slot.layer_idx == layer_idx && slot.expert_idx == eid) {
                    already_resident = true;
                    break;
                }
            }
            if (already_resident) {
                diagnostic_stats_.prefetch_reject_resident++;
                continue;
            }

            ids.push_back(eid);
        }
    }

    if (ids.empty()) {
        return false;
    }

    const int admitted_count = int(ids.size());
    try {
        std::future<bool> future = std::async(
                std::launch::async,
                [this, layer_idx, ids = std::move(ids)]() mutable {
                    try {
                        return this->prefetch_experts(layer_idx, ids.data(), int(ids.size()));
                    } catch (...) {
                        return false;
                    }
                });
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        async_prefetches_.push_back({ layer_idx, admitted_count, { layer_idx }, std::move(future) });
        diagnostic_stats_.prefetch_submitted_calls++;
        diagnostic_stats_.prefetch_submitted_experts += uint64_t(admitted_count);
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG prefetch:force_async_start layer=%d ids=%d pending=%zu\n",
                layer_idx, admitted_count, async_prefetches_.size());
        return true;
    } catch (...) {
        return false;
    }
}

bool DStorageSlotManager::enqueue_global_prefetch(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected) {
    if (!llama_dstorage_global_prefetch_enabled() || expert_ids == nullptr || n_selected <= 0) {
        return false;
    }

    std::vector<int32_t> ids;
    {
        std::lock_guard<std::mutex> slot_lock(slots_mutex_);
        diagnostic_stats_.prefetch_candidate_calls++;
        const bool high_confidence_prefetch = llama_dstorage_high_confidence_prefetch_enabled();
        const uint32_t observations = high_confidence_prefetch
                ? request_activation_stats_.rolling_decode_observations(layer_idx)
                : request_activation_stats_.decode_observations(layer_idx);
        const bool allow_route_prediction = llama_dstorage_predictive_prefetch_enabled();
        if ((high_confidence_prefetch || !allow_route_prediction) &&
                observations < uint32_t(llama_dstorage_prefetch_min_observations())) {
            diagnostic_stats_.prefetch_reject_observations += uint64_t(n_selected);
            return false;
        }

        bool has_free_slot = false;
        int64_t victim_retention_cost = std::numeric_limits<int64_t>::max();
        const int speculative_slots = speculative_slot_count();
        for (int slot_idx = 0; slot_idx < int(slots_.size()); ++slot_idx) {
            if (!slot_in_speculative_partition(slot_idx, speculative_slots)) {
                continue;
            }
            const ExpertSlot & slot = slots_[slot_idx];
            if (!slot.occupied) {
                has_free_slot = true;
                victim_retention_cost = 0;
                break;
            }
            if (slot_is_protected(slot)) {
                continue;
            }
            victim_retention_cost = std::min(
                    victim_retention_cost,
                    calculate_retention_score(
                            slot,
                            request_activation_stats_.count(
                                    slot.layer_idx,
                                    slot.expert_idx,
                                    dstorage_moe_phase::decode),
                            request_activation_stats_.count(
                                    slot.layer_idx,
                                    slot.expert_idx,
                                    dstorage_moe_phase::prefill),
                            layer_idx,
                            n_layers_));
        }
        diagnostic_stats_.prefetch_lowest_victim_cost = std::min(
                diagnostic_stats_.prefetch_lowest_victim_cost,
                victim_retention_cost);

        ids.reserve(n_selected);
        for (int i = 0; i < n_selected; ++i) {
            const int32_t eid = expert_ids[i];
            if (eid < 0 || std::find(ids.begin(), ids.end(), eid) != ids.end()) {
                continue;
            }

            bool already_resident = false;
            for (const ExpertSlot & slot : slots_) {
                if (slot.occupied && slot.layer_idx == layer_idx && slot.expert_idx == eid) {
                    already_resident = true;
                    break;
                }
            }
            if (already_resident) {
                diagnostic_stats_.prefetch_reject_resident++;
                continue;
            }

            const double min_confidence = llama_dstorage_prefetch_min_confidence();
            if (high_confidence_prefetch &&
                    observations < uint32_t(llama_dstorage_prefetch_min_observations())) {
                diagnostic_stats_.prefetch_reject_observations++;
                continue;
            }
            const double confidence = observations > 0
                    ? (high_confidence_prefetch
                            ? request_activation_stats_.rolling_decode_confidence(layer_idx, eid)
                            : request_activation_stats_.decode_confidence(layer_idx, eid))
                    : (high_confidence_prefetch ? 0.0 : min_confidence);
            diagnostic_stats_.prefetch_best_confidence =
                    std::max(diagnostic_stats_.prefetch_best_confidence, confidence);
            ExpertSlot candidate;
            candidate.layer_idx = layer_idx;
            candidate.expert_idx = eid;
            candidate.prefetch_hits = 1000;
            const uint32_t candidate_decode_count = high_confidence_prefetch
                    ? request_activation_stats_.rolling_decode_count(layer_idx, eid)
                    : request_activation_stats_.count(
                            layer_idx,
                            eid,
                            dstorage_moe_phase::decode);
            const int64_t candidate_value = calculate_retention_score(
                    candidate,
                    candidate_decode_count,
                    request_activation_stats_.count(
                            layer_idx,
                            eid,
                            dstorage_moe_phase::prefill),
                    layer_idx,
                    n_layers_);
            diagnostic_stats_.prefetch_best_candidate_value = std::max(
                    diagnostic_stats_.prefetch_best_candidate_value,
                    candidate_value);

            if (should_admit_prefetch(
                        confidence,
                        candidate_value,
                        victim_retention_cost,
                        has_free_slot,
                        min_confidence)) {
                ids.push_back(eid);
            } else {
                diagnostic_stats_.prefetch_reject_policy++;
            }
        }
    }
    if (ids.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
    for (QueuedGlobalPrefetch & queued : global_prefetch_queue_) {
        if (queued.layer_idx != layer_idx) {
            continue;
        }
        for (int32_t eid : ids) {
            if (std::find(queued.expert_ids.begin(), queued.expert_ids.end(), eid) == queued.expert_ids.end()) {
                queued.expert_ids.push_back(eid);
            }
        }
        return true;
    }
    global_prefetch_queue_.push_back({ layer_idx, std::move(ids) });
    return true;
}

bool DStorageSlotManager::flush_global_prefetch_async() {
    if (!llama_dstorage_global_prefetch_enabled()) {
        return false;
    }

    collect_async_prefetches(false);

    std::vector<QueuedGlobalPrefetch> batch;
    {
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        if (global_prefetch_queue_.empty()) {
            return false;
        }
        batch.swap(global_prefetch_queue_);
    }

    int total_experts = 0;
    std::vector<int> affected_layers;
    affected_layers.reserve(batch.size());
    for (const QueuedGlobalPrefetch & request : batch) {
        total_experts += int(request.expert_ids.size());
        if (std::find(affected_layers.begin(), affected_layers.end(), request.layer_idx) == affected_layers.end()) {
            affected_layers.push_back(request.layer_idx);
        }
    }
    if (total_experts <= 0) {
        return false;
    }

    try {
        std::future<bool> future = std::async(
                std::launch::async,
                [this, batch = std::move(batch)]() mutable {
                    try {
                        return this->prefetch_global_batch(std::move(batch));
                    } catch (...) {
                        return false;
                    }
                });
        {
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            diagnostic_stats_.prefetch_submitted_calls++;
            diagnostic_stats_.prefetch_submitted_experts += uint64_t(total_experts);
        }
        size_t pending_count = 0;
        const size_t layer_count = affected_layers.size();
        {
            std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
            async_prefetches_.push_back({ -1, total_experts, std::move(affected_layers), std::move(future) });
            pending_count = async_prefetches_.size();
        }
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG prefetch:global_async_start layers=%zu ids=%d pending=%zu\n",
                layer_count,
                total_experts,
                pending_count);
        return true;
    } catch (...) {
        return false;
    }
}

void DStorageSlotManager::wait_for_layer_prefetch(int layer_idx) {
    std::future<bool> target_future;
    bool found = false;

    {
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        for (auto it = async_prefetches_.begin(); it != async_prefetches_.end(); ++it) {
            const bool affects_layer =
                    it->layer_idx == layer_idx ||
                    std::find(it->affected_layers.begin(), it->affected_layers.end(), layer_idx) != it->affected_layers.end();
            if (affects_layer) {
                target_future = std::move(it->future);
                async_prefetches_.erase(it);
                found = true;
                break;
            }
        }
    }

    if (found) {
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots: waiting for L%d speculative prefetch\n", layer_idx);
        const int64_t t_start = llama_dstorage_now_us();
        bool ok = target_future.get();
        if (ok && ds_loader_ != nullptr) {
            ds_loader_cuda_wait_event(ds_loader_);
        }
        const int64_t t_end = llama_dstorage_now_us();
        {
            std::lock_guard<std::mutex> slot_lock(slots_mutex_);
            diagnostic_stats_.prefetch_wait_calls++;
            diagnostic_stats_.prefetch_wait_us += uint64_t(std::max<int64_t>(0, t_end - t_start));
        }
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots: L%d speculative prefetch wait finished (ok=%d, elapsed_us=%lld)\n", 
                layer_idx, ok ? 1 : 0, (long long)(t_end - t_start));
    }
}

bool DStorageSlotManager::ensure_experts_loaded(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected,
        std::unordered_map<std::string, uint64_t> & out_pool_ptrs,
        std::vector<int32_t> & out_id_map,
        dstorage_moe_phase phase) {

    const int64_t t_total0 = llama_dstorage_now_us();
    uint64_t trace_stream_file_bytes = 0;
    uint64_t trace_transfer_us = 0;
    if (n_selected <= 0) {
        return true; // No experts to load, success
    }
    if (expert_ids == nullptr) {
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_loaded invalid input (null ids) layer=%d n_selected=%d\n",
                layer_idx, n_selected);
        return false;
    }

    int64_t t_step = llama_dstorage_now_us();
    if (phase == dstorage_moe_phase::decode) {
        wait_for_layer_prefetch(layer_idx);
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "wait_layer_prefetch", t_step, llama_dstorage_now_us());

    t_step = llama_dstorage_now_us();
    std::unique_lock<std::mutex> slot_lock(slots_mutex_);
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "acquire_slot_lock", t_step, llama_dstorage_now_us());

    if (phase != dstorage_moe_phase::prefetch) {
        const bool starts_request =
                !request_tracking_started_ ||
                (phase == dstorage_moe_phase::prefill &&
                 last_request_phase_ == dstorage_moe_phase::decode);
        if (starts_request) {
            request_activation_stats_.begin_request();
            request_tracking_started_ = true;
        }
        const bool decode_like_for_history =
                phase == dstorage_moe_phase::prefill &&
                n_expert_used_ > 0 &&
                n_selected <= n_expert_used_ * 4;
        const int64_t t_ras = llama_dstorage_now_us();
        request_activation_stats_.record(
                layer_idx,
                expert_ids,
                n_selected,
                phase,
                decode_like_for_history);
        llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "activation_stats_record", t_ras, llama_dstorage_now_us());
        last_request_phase_ = phase;
    }

    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_loaded enter layer=%d n_selected=%d phase=%s phase_policy=%d pools_allocated=%d\n",
            layer_idx, n_selected, llama_dstorage_phase_name(phase), phase_cache_policy_ ? 1 : 0, pools_allocated_ ? 1 : 0);

    const bool is_async_prefetch = (phase == dstorage_moe_phase::prefetch);
    const dstorage_moe_phase target_phase =
            phase == dstorage_moe_phase::prefill
                    ? dstorage_moe_phase::prefill
                    : dstorage_moe_phase::decode;
    const bool need_reallocate = !is_async_prefetch && should_reallocate_for_phase(
            phase_cache_policy_,
            current_allocated_phase_,
            target_phase,
            pool_budget_bytes_prefill_,
            pool_budget_bytes_decode_);

    if (!is_async_prefetch && current_allocated_phase_ != target_phase) {
        pool_budget_bytes_ = target_phase == dstorage_moe_phase::prefill
                ? pool_budget_bytes_prefill_
                : pool_budget_bytes_decode_;
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG slots: phase transition %s -> %s, budget %.2f MiB, reallocate=%d\n",
                llama_dstorage_phase_name(current_allocated_phase_),
                llama_dstorage_phase_name(target_phase),
                pool_budget_bytes_ / 1024.0 / 1024.0,
                need_reallocate ? 1 : 0);
        current_allocated_phase_ = target_phase;
    }

    if (need_reallocate && pools_allocated_) {
        // Safely wait for all outstanding prefetches and admissions
        slot_lock.unlock();
        collect_async_prefetches(true);
        collect_pinned_admissions(true);
        slot_lock.lock();

#if LLAMA_DSTORAGE_HAS_BACKEND
        for (auto & [key, pool] : type_pools_) {
            if (pool.cuda_ptr != 0 && pool.owns_alloc) {
                ds_loader_cuda_free(pool.alloc_ptr != 0 ? pool.alloc_ptr : pool.cuda_ptr);
                pool.cuda_ptr = 0;
                pool.alloc_ptr = 0;
            }
        }
        for (auto & [key, pool] : active_type_pools_) {
            if (pool.cuda_ptr != 0 && pool.owns_alloc) {
                ds_loader_cuda_free(pool.alloc_ptr != 0 ? pool.alloc_ptr : pool.cuda_ptr);
                pool.cuda_ptr = 0;
                pool.alloc_ptr = 0;
            }
        }
        if (bundle_staging_ptr_ != 0) {
            ds_loader_cuda_free(bundle_staging_ptr_);
            bundle_staging_ptr_ = 0;
            bundle_staging_size_ = 0;
        }
#endif
        type_pools_.clear();
        active_type_pools_.clear();
        active_capacity_ = 0;
        active_slot_count_ = 0;
        execution_slots_by_layer_.clear();
        slots_.clear();
        pool_group_slot_counts_.clear();
        layer_pool_groups_.clear();
        resident_experts_.clear();
        reset_arc_state();
        n_slots_ = 0;
        pools_allocated_ = false;
    }

    int64_t t_pools = llama_dstorage_now_us();
    if (!ensure_pools_allocated()) {
        return false;
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "ensure_pools_call", t_pools, llama_dstorage_now_us());
    t_pools = llama_dstorage_now_us();
    if (!preload_static_pinned_layers()) {
        return false;
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "preload_static_pinned", t_pools, llama_dstorage_now_us());
    t_pools = llama_dstorage_now_us();
    if (phase == dstorage_moe_phase::decode) {
        release_prefill_only_pinned_cache();
    }
    collect_pinned_admissions(false);
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "release_and_collect", t_pools, llama_dstorage_now_us());
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "ensure_pools_allocated", t_total0, llama_dstorage_now_us());
    const bool mtp_decode_like_accounting =
            llama_dstorage_mtp_decode_phase_enabled() &&
            phase == dstorage_moe_phase::prefill &&
            n_expert_used_ > 0 &&
            n_selected <= n_expert_used_ * 4;
    const dstorage_moe_phase accounting_phase =
            mtp_decode_like_accounting ? dstorage_moe_phase::decode : phase;
    PhaseStats & phase_stats =
            accounting_phase == dstorage_moe_phase::decode   ? decode_stats_ :
            accounting_phase == dstorage_moe_phase::prefetch ? prefetch_stats_ :
                                                                prefill_stats_;
    phase_stats.calls++;
    if (accounting_phase == dstorage_moe_phase::decode && n_layers_ > 0 && decode_stats_.calls % (n_layers_ * 4) == 0) {
        for (int s = 0; s < n_slots_; s++) {
            if (slots_[s].occupied) {
                slots_[s].decode_hits = (slots_[s].decode_hits * 95) / 100;
            }
        }
    }

    // --- Build unique, valid expert list ---
    int64_t t0 = llama_dstorage_now_us();
    std::vector<int32_t> unique_experts;
    for (int i = 0; i < n_selected; i++) {
        if (expert_ids[i] >= 0) {
            bool found = false;
            for (int32_t e : unique_experts) {
                if (e == expert_ids[i]) { found = true; break; }
            }
            if (!found) {
                unique_experts.push_back(expert_ids[i]);
            }
        }
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "build_unique_expert_list", t0, llama_dstorage_now_us());

    if ((int)unique_experts.size() > n_slots_) {
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_loaded capacity_fail layer=%d unique=%zu n_slots=%d\n",
                layer_idx, unique_experts.size(), n_slots_);
        LLAMA_LOG_ERROR("%s: need %d unique experts but only have %d slots (n_selected=%d)\n",
                __func__, (int)unique_experts.size(), n_slots_, n_selected);
        return false;
    }

    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_loaded unique=%zu n_slots=%d\n",
            unique_experts.size(), n_slots_);

    const bool decode_hot_accounting =
            accounting_phase == dstorage_moe_phase::decode &&
            (phase == dstorage_moe_phase::decode ||
             llama_dstorage_decode_hot_prefill_like_enabled());
    const bool pinned_decode_like =
            phase != dstorage_moe_phase::prefetch &&
            n_expert_used_ > 0 &&
            n_selected <= n_expert_used_ * 4;

    if (pinned_cache_enabled()) {
        for (int32_t eid : unique_experts) {
            pinned_frequency_[make_expert_key(layer_idx, eid)]++;
        }
    }

    // --- Classify each unique expert as cache-hit or cache-miss ---
    t0 = llama_dstorage_now_us();
    std::vector<int32_t> to_load;
    std::unordered_map<int32_t, int> expert_to_slot; // expert_id -> slot index

    for (int32_t eid : unique_experts) {
        bool hit = false;
        for (int s = 0; s < n_slots_; s++) {
            if (slots_[s].occupied &&
                slots_[s].layer_idx == layer_idx &&
                slots_[s].expert_idx == eid) {
                expert_to_slot[eid] = s;
                if (phase != dstorage_moe_phase::prefetch && slots_[s].prefetch_hits > 0) {
                    diagnostic_stats_.useful_prefetch_hits++;
                    if (accounting_phase == dstorage_moe_phase::decode) {
                        diagnostic_stats_.useful_prefetch_decode_hits++;
                    } else {
                        diagnostic_stats_.useful_prefetch_prefill_hits++;
                    }
                }
                slots_[s].last_used_tick = ++use_tick_;
                arc_record_hit(slots_[s], accounting_phase);
                if (accounting_phase == dstorage_moe_phase::decode) {
                    slots_[s].decode_hits += 1000;
                } else if (accounting_phase == dstorage_moe_phase::prefetch) {
                    slots_[s].prefetch_hits += 1000;
                } else {
                    slots_[s].prefill_hits += 1000;
                }
                hit = true;
                break;
            }
        }
        if (!hit) {
            const uint64_t expert_key = make_expert_key(layer_idx, eid);
            const bool is_pinned_hit = pinned_decode_like && pinned_cache_enabled() && pinned_cache_.find(expert_key) != pinned_cache_.end();
            if (is_pinned_hit) {
                // A pinned hit is only an L2 hit. It still needs a VRAM slot
                // and a host->device copy before MUL_MAT_ID can consume it.
                pinned_cache_[expert_key].last_used_tick = ++pinned_tick_;
            }
        }
        if (llama_dstorage_decode_hot_cache_enabled() &&
                decode_hot_accounting) {
            const uint64_t expert_key = make_expert_key(layer_idx, eid);
            const bool was_protected = decode_hot_protected(expert_key);
            decode_hot_stats_.accesses++;
            decode_hot_counts_[expert_key]++;
            decode_hot_last_seen_[expert_key] = ++decode_hot_tick_;
            if (hit) {
                decode_hot_stats_.hits++;
                if (was_protected) {
                    decode_hot_stats_.protected_hits++;
                }
            } else {
                decode_hot_stats_.misses++;
                decode_hot_misses_[expert_key]++;
                if (was_protected) {
                    decode_hot_stats_.protected_misses++;
                }
            }
        }
        if (!hit) {
            to_load.push_back(eid);
        }
    }
    if (llama_dstorage_decode_hot_cache_enabled() &&
            decode_hot_accounting) {
        const int rebuild_interval = llama_dstorage_decode_hot_rebuild_interval();
        if (decode_hot_protected_.empty() ||
                (decode_hot_stats_.accesses % uint64_t(rebuild_interval)) == 0) {
            rebuild_decode_hot_protected_set();
        }
    }
    phase_stats.hits += unique_experts.size() - to_load.size();
    phase_stats.misses += to_load.size();
    if (to_load.empty()) {
        phase_stats.all_hit_calls++;
    } else {
        phase_stats.miss_calls++;
    }
    if (phase == dstorage_moe_phase::prefetch) {
        diagnostic_stats_.prefetch_loaded_experts += uint64_t(to_load.size());
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "classify_cache_hits_misses", t0, llama_dstorage_now_us());

    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_loaded phase=%s accounting=%s hits=%zu misses=%zu layer_tensors_pending=%zu totals_prefill(h=%" PRIu64 " m=%" PRIu64 " e_prefill=%" PRIu64 " e_decode=%" PRIu64 ") totals_decode(h=%" PRIu64 " m=%" PRIu64 " e_prefill=%" PRIu64 " e_decode=%" PRIu64 ") totals_prefetch(h=%" PRIu64 " m=%" PRIu64 " e_prefill=%" PRIu64 " e_decode=%" PRIu64 ")\n",
            llama_dstorage_phase_name(phase), llama_dstorage_phase_name(accounting_phase),
            unique_experts.size() - to_load.size(), to_load.size(), layer_tensors_.count(layer_idx),
            prefill_stats_.hits, prefill_stats_.misses, prefill_stats_.evicted_prefill_only, prefill_stats_.evicted_decode_touched,
            decode_stats_.hits, decode_stats_.misses, decode_stats_.evicted_prefill_only, decode_stats_.evicted_decode_touched,
            prefetch_stats_.hits, prefetch_stats_.misses, prefetch_stats_.evicted_prefill_only, prefetch_stats_.evicted_decode_touched);

    // Stream misses in expert-id order so adjacent GGUF slices can be coalesced
    // when slot allocation also lands in adjacent global slots.
    t0 = llama_dstorage_now_us();
    std::sort(to_load.begin(), to_load.end());
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "sort_misses", t0, llama_dstorage_now_us());

    // --- Locate the layer's tensor list (needed for both miss and output building) ---
    auto layer_it = layer_tensors_.find(layer_idx);
    if (layer_it == layer_tensors_.end()) {
        LLAMA_LOG_ERROR("%s: no tensors registered for layer %d\n", __func__, layer_idx);
        return false;
    }
    const std::vector<std::string> & layer_tnames = layer_it->second;
    LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:ensure_loaded layer=%d tensor_count=%zu\n",
            layer_idx, layer_tnames.size());

    struct LoadPlan {
        int32_t eid = -1;
        int target_slot = -1;
        bool from_pinned = false;
    };

    std::vector<LoadPlan> load_plan;
    load_plan.reserve(to_load.size());
    std::vector<int32_t> cold_to_load;
    cold_to_load.reserve(to_load.size());
    std::vector<int32_t> pinned_to_load;
    pinned_to_load.reserve(to_load.size());
    const int speculative_slots = speculative_slot_count();
    auto slot_allowed_for_phase = [&](int slot_idx) {
        const bool speculative_slot = slot_in_speculative_partition(
                slot_idx, speculative_slots);
        if (!slot_compatible_with_layer(slot_idx, layer_idx)) {
            return false;
        }
        return phase == dstorage_moe_phase::prefetch ? speculative_slot : !speculative_slot;
    };

    auto slot_unavailable_for_load = [&](int slot_idx) {
        if (!slot_allowed_for_phase(slot_idx)) {
            return true;
        }
        // A speculative next-layer prefetch can overlap graph execution, so it
        // must not evict the layer that spawned it or the target layer if the
        // actual router has already loaded some of its experts.
        if (phase == dstorage_moe_phase::prefetch &&
                !llama_dstorage_relax_prefetch_layer_guard() &&
                slots_[slot_idx].occupied &&
                (slots_[slot_idx].layer_idx == layer_idx - 1 ||
                 slots_[slot_idx].layer_idx == layer_idx)) {
            return true;
        }
        for (const auto & [k, v] : expert_to_slot) {
            GGML_UNUSED(k);
            if (v == slot_idx) { return true; }
        }
        for (const LoadPlan & plan : load_plan) {
            if (plan.target_slot == slot_idx) { return true; }
        }
        return slot_is_protected(slots_[slot_idx]);
    };

    auto choose_eviction_slot = [&]() {
        int target_slot = -1;
        for (int s = 0; s < n_slots_; s++) {
            if (slot_unavailable_for_load(s)) {
                continue;
            }
            if (target_slot < 0 ||
                    is_better_request_eviction_candidate(
                            slots_[s],
                            slots_[target_slot],
                            layer_idx)) {
                target_slot = s;
            }
        }
        return target_slot;
    };

    // --- Reserve one slot for every cache-miss expert before streaming ---
    t0 = llama_dstorage_now_us();
    const bool contiguous_load_slots = llama_dstorage_contiguous_load_slots();
    for (int32_t eid : to_load) {
        const uint64_t expert_key = make_expert_key(layer_idx, eid);

        int target_slot = -1;

        if (contiguous_load_slots && !load_plan.empty()) {
            const LoadPlan & previous = load_plan.back();
            const int adjacent_slot = previous.target_slot + 1;
            if (previous.eid + 1 == eid &&
                    adjacent_slot >= 0 &&
                    adjacent_slot < n_slots_ &&
                    !slot_unavailable_for_load(adjacent_slot)) {
                target_slot = adjacent_slot;
            }
        }

        // Find a free slot
        for (int s = 0; s < n_slots_; s++) {
            if (target_slot >= 0) {
                break;
            }
            bool reserved = false;
            for (const LoadPlan & plan : load_plan) {
                if (plan.target_slot == s) { reserved = true; break; }
            }
            if (!reserved &&
                    slot_allowed_for_phase(s) &&
                    !slots_[s].occupied &&
                    !slots_[s].admission_pending) {
                target_slot = s;
                break;
            }
        }

        // If no free slot, evict the least recently used slot not needed by the current batch.
        if (target_slot < 0) {
            target_slot = choose_eviction_slot();
        }

        if (target_slot < 0) {
            collect_pinned_admissions(true);
            for (int s = 0; s < n_slots_; s++) {
                bool reserved = false;
                for (const LoadPlan & plan : load_plan) {
                    if (plan.target_slot == s) { reserved = true; break; }
                }
                if (!reserved &&
                        slot_allowed_for_phase(s) &&
                        !slots_[s].occupied &&
                        !slots_[s].admission_pending) {
                    target_slot = s;
                    break;
                }
            }
            if (target_slot < 0) {
                target_slot = choose_eviction_slot();
            }
            if (target_slot < 0) {
                // Try recovering stale transfer_pending slots from previous failed calls
                int stale_cleared = 0;
                for (int s = 0; s < n_slots_; s++) {
                    if (slots_[s].transfer_pending && !slots_[s].occupied) {
                        slots_[s].transfer_pending = false;
                        stale_cleared++;
                    }
                }
                if (stale_cleared > 0) {
                    for (int s = 0; s < n_slots_; s++) {
                        bool reserved = false;
                        for (const LoadPlan & p : load_plan) {
                            if (p.target_slot == s) { reserved = true; break; }
                        }
                        if (!reserved && slot_allowed_for_phase(s) && !slots_[s].occupied && !slots_[s].admission_pending) {
                            target_slot = s;
                            break;
                        }
                    }
                    if (target_slot >= 0) {
                        LLAMA_LOG_WARN("%s: recovered slot %d after clearing %d stale transfer_pending\n",
                                __func__, target_slot, stale_cleared);
                    }
                }
            }
            if (target_slot < 0) {
                LLAMA_LOG_ERROR("%s: no available slot for expert %d\n", __func__, eid);
                return false;
            }
        }

        if (slots_[target_slot].occupied) {
            if (decode_hot_protected(slots_[target_slot])) {
                decode_hot_stats_.protected_evictions++;
            }
            if (slots_[target_slot].prefetch_hits > 0) {
                if (slots_[target_slot].decode_hits == 0 && slots_[target_slot].prefill_hits == 0) {
                    diagnostic_stats_.prefetch_evicted_before_use++;
                } else {
                    diagnostic_stats_.prefetch_evicted_after_use++;
                }
            }
            arc_record_eviction(slots_[target_slot]);
            if (slots_[target_slot].decode_hits > 0) {
                phase_stats.evicted_decode_touched++;
            } else {
                phase_stats.evicted_prefill_only++;
            }
            LLAMA_DSTORAGE_DEBUG_LOG(
                    "DStorage DEBUG slots:evict phase=%s policy=%d victim_slot=%d old=(L%d,E%d) decode_hits=%u prefill_hits=%u tick=%" PRIu64 "\n",
                    llama_dstorage_phase_name(phase), phase_cache_policy_ ? 1 : 0,
                    target_slot, slots_[target_slot].layer_idx, slots_[target_slot].expert_idx,
                    slots_[target_slot].decode_hits, slots_[target_slot].prefill_hits,
                    slots_[target_slot].last_used_tick);
        }

        slots_[target_slot].layer_idx  = -1;
        slots_[target_slot].expert_idx = -1;
        slots_[target_slot].occupied   = false;
        slots_[target_slot].transfer_pending = true;
        slots_[target_slot].execution_pins = 0;
        slots_[target_slot].prefill_hits = 0;
        slots_[target_slot].decode_hits = 0;
        slots_[target_slot].prefetch_hits = 0;
        slots_[target_slot].arc_segment = dstorage_arc_segment::none;
        slots_[target_slot].contiguous_neighbors = 0;

        const bool pinned_hit = pinned_cache_enabled() && pinned_cache_.find(expert_key) != pinned_cache_.end();
        if (pinned_hit) {
            pinned_cache_[expert_key].last_used_tick = ++pinned_tick_;
            pinned_to_load.push_back(eid);
        } else {
            cold_to_load.push_back(eid);
        }
        load_plan.push_back({ eid, target_slot, pinned_hit });
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "reserve_or_evict_slots", t0, llama_dstorage_now_us());
    LLAMA_DSTORAGE_DEBUG_LOG(
            "DStorage DEBUG slots:pinned layer=%d enabled=%d hits=%zu cold=%zu entries=%zu used_mib=%.2f budget_mib=%.2f\n",
            layer_idx, pinned_cache_enabled() ? 1 : 0, pinned_to_load.size(), cold_to_load.size(),
            pinned_cache_.size(), pinned_used_bytes_ / 1024.0 / 1024.0, pinned_budget_bytes_ / 1024.0 / 1024.0);

    // --- Batch stream every registered tensor type for every cache-miss expert ---
#if LLAMA_DSTORAGE_HAS_BACKEND
    t0 = llama_dstorage_now_us();
    std::vector<DSLoaderStreamRequest> batch_requests;
    batch_requests.reserve(load_plan.size() * layer_tnames.size());
    const int max_coalesced_experts = llama_dstorage_max_coalesced_experts();

    struct CoalescedRun {
        int32_t first_eid = -1;
        int32_t last_eid = -1;
        int first_slot = -1;
        int last_slot = -1;
    };

    std::vector<CoalescedRun> runs;
    runs.reserve(load_plan.size());
    uint64_t possible_run_count = 0;
    uint64_t possible_run_expert_count = 0;
    int32_t previous_possible_eid = -1;
    for (const LoadPlan & plan : load_plan) {
        if (plan.from_pinned) {
            continue;
        }
        if (possible_run_count == 0 || plan.eid != previous_possible_eid + 1) {
            possible_run_count++;
        }
        possible_run_expert_count++;
        previous_possible_eid = plan.eid;

        if (!runs.empty()) {
            CoalescedRun & run = runs.back();
            const bool can_extend =
                    (run.last_eid - run.first_eid + 1) < max_coalesced_experts &&
                    plan.eid == run.last_eid + 1 &&
                    plan.target_slot == run.last_slot + 1;
            if (can_extend) {
                run.last_eid = plan.eid;
                run.last_slot = plan.target_slot;
                continue;
            }
        }
        runs.push_back({ plan.eid, plan.eid, plan.target_slot, plan.target_slot });
    }
    uint64_t run_expert_count = 0;
    for (const CoalescedRun & run : runs) {
        run_expert_count += uint64_t(run.last_eid - run.first_eid + 1);
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "build_coalesced_runs", t0, llama_dstorage_now_us());
#endif

#if LLAMA_DSTORAGE_HAS_BACKEND
    t0 = llama_dstorage_now_us();
    std::vector<DSLoaderHostToCudaRequest> pinned_requests;
    uint64_t pinned_h2d_bytes = 0;
    for (const LoadPlan & plan : load_plan) {
        if (!plan.from_pinned) {
            continue;
        }
        const uint64_t expert_key = make_expert_key(layer_idx, plan.eid);
        auto pin_it = pinned_cache_.find(expert_key);
        if (pin_it == pinned_cache_.end()) {
            LLAMA_LOG_ERROR("%s: pinned-cache entry disappeared for layer %d expert %d\n",
                    __func__, layer_idx, plan.eid);
            return false;
        }

        const PinnedEntry & entry = pin_it->second;
        for (const PinnedSlice & slice : entry.slices) {
            const std::string pool_key = pool_key_for_layer(layer_idx, slice.tensor_name);
            auto pool_it = type_pools_.find(pool_key);
            if (pool_it == type_pools_.end()) {
                LLAMA_LOG_ERROR("%s: no VRAM pool for pinned tensor type '%s'\n",
                        __func__, pool_key.c_str());
                return false;
            }
            const TensorTypePool & pool = pool_it->second;
            const uint64_t dest = pool.cuda_ptr + uint64_t(slots_[plan.target_slot].pool_index) * pool.slot_size;
            const char * src = reinterpret_cast<const char *>(entry.host_ptr) + slice.offset;
            pinned_requests.push_back({ src, slice.size, dest });
            pinned_h2d_bytes += slice.size;
        }
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "build_pinned_h2d_requests", t0, llama_dstorage_now_us());

    if (!pinned_requests.empty()) {
        t0 = llama_dstorage_now_us();
        const int result = ds_loader_host_to_cuda_batch(
                ds_loader_,
                pinned_requests.data(),
                int(pinned_requests.size()));
        const uint64_t pinned_h2d_us = uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - t0));
        llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "pinned_host_to_cuda_batch", t0, t0 + int64_t(pinned_h2d_us));
        if (result != 0) {
            LLAMA_LOG_ERROR("DirectStorage: failed pinned host->GPU load L%d (%zu requests), hr=0x%08" PRIx32 "\n",
                    layer_idx, pinned_requests.size(), uint32_t(ds_loader_get_hresult()));
            return false;
        }

        if (phase != dstorage_moe_phase::prefetch) {
            ds_loader_cuda_wait_event(ds_loader_);
        }
        phase_stats.stream_pinned_bytes += pinned_h2d_bytes;
        phase_stats.stream_pinned_us += pinned_h2d_us;
        if (batch_requests.empty()) {
            phase_stats.stream_pinned_experts += uint64_t(pinned_to_load.size());
        }
    }

    t0 = llama_dstorage_now_us();
    for (const CoalescedRun & run : runs) {
        if (llama_dstorage_fused_expert_pools() && !layer_tnames.empty()) {
            bool can_read_fused_record = true;
            const std::string & first_tname = layer_tnames.front();
            const std::string first_tkey = make_tensor_key(layer_idx, first_tname.c_str());
            auto first_reg_it = tensor_registry_.find(first_tkey);
            uint64_t record_file_base = 0;
            uint64_t record_stride = 0;
            uint64_t fused_cuda_base = 0;
            if (first_reg_it == tensor_registry_.end() ||
                    !first_reg_it->second.sidecar_expert_major ||
                    first_reg_it->second.total_size_compressed > 0) {
                can_read_fused_record = false;
            } else {
                const ExpertTensorInfo & first_info = first_reg_it->second;
                const std::string first_pool_key = pool_key_for_layer(layer_idx, first_tname);
                auto first_pool_it = type_pools_.find(first_pool_key);
                if (first_pool_it == type_pools_.end() ||
                        first_pool_it->second.slot_size != first_info.sidecar_record_stride ||
                        first_pool_it->second.cuda_ptr < first_info.sidecar_tensor_offset) {
                    can_read_fused_record = false;
                } else {
                    record_file_base = first_info.file_offset;
                    record_stride = first_info.sidecar_record_stride;
                    fused_cuda_base = first_pool_it->second.cuda_ptr - first_info.sidecar_tensor_offset;
                }
            }

            for (int i = 0; can_read_fused_record && i < (run.last_eid - run.first_eid + 1); ++i) {
                const int slot = run.first_slot + i;
                if (slot < 0 ||
                        slot >= int(slots_.size()) ||
                        slots_[slot].pool_index != slots_[run.first_slot].pool_index + i) {
                    can_read_fused_record = false;
                    break;
                }
            }

            for (const std::string & tname : layer_tnames) {
                if (!can_read_fused_record) {
                    break;
                }
                const std::string tkey = make_tensor_key(layer_idx, tname.c_str());
                auto reg_it = tensor_registry_.find(tkey);
                const std::string pool_key = pool_key_for_layer(layer_idx, tname);
                auto pool_it = type_pools_.find(pool_key);
                if (reg_it == tensor_registry_.end() ||
                        pool_it == type_pools_.end() ||
                        !reg_it->second.sidecar_expert_major ||
                        reg_it->second.total_size_compressed > 0 ||
                        reg_it->second.file_path != first_reg_it->second.file_path ||
                        reg_it->second.file_offset != record_file_base ||
                        reg_it->second.sidecar_record_stride != record_stride ||
                        pool_it->second.slot_size != record_stride ||
                        pool_it->second.cuda_ptr != fused_cuda_base + reg_it->second.sidecar_tensor_offset) {
                    can_read_fused_record = false;
                    break;
                }
            }

            if (can_read_fused_record) {
                const int run_experts = run.last_eid - run.first_eid + 1;
                DSLoaderStreamRequest req = {};
                req.file_path = first_reg_it->second.file_path.c_str();
                req.file_offset = record_file_base + uint64_t(run.first_eid) * record_stride;
                req.size = uint64_t(run_experts) * record_stride;
                req.cuda_dest_ptr = fused_cuda_base + uint64_t(slots_[run.first_slot].pool_index) * record_stride;
                req.uncompressed_size = req.size;
                batch_requests.push_back(req);
                LLAMA_DSTORAGE_DEBUG_LOG(
                        "DStorage DEBUG slots:fused enqueue layer=%d experts=%d..%d slots=%d..%d bytes=%" PRIu64 " file_off=%" PRIu64 " dest=0x%" PRIx64 "\n",
                        layer_idx, run.first_eid, run.last_eid, run.first_slot, run.last_slot,
                        req.size, req.file_offset, req.cuda_dest_ptr);
                continue;
            }
        }

        for (const std::string & tname : layer_tnames) {
            const std::string tkey   = make_tensor_key(layer_idx, tname.c_str());
            auto reg_it = tensor_registry_.find(tkey);
            if (reg_it == tensor_registry_.end()) {
                LLAMA_LOG_ERROR("%s: tensor '%s' not found in registry for layer %d\n",
                        __func__, tname.c_str(), layer_idx);
                return false;
            }
            const ExpertTensorInfo & info = reg_it->second;

            const std::string pool_key = pool_key_for_layer(layer_idx, tname);
            auto pool_it = type_pools_.find(pool_key);
            if (pool_it == type_pools_.end()) {
                LLAMA_LOG_ERROR("%s: no VRAM pool for tensor type '%s'\n",
                        __func__, pool_key.c_str());
                return false;
            }
            const TensorTypePool & pool = pool_it->second;

            const int run_experts = run.last_eid - run.first_eid + 1;
            if (info.total_size_compressed > 0) {
                // Decompression path: each expert must be loaded individually
                for (int i = 0; i < run_experts; ++i) {
                    const int32_t eid = run.first_eid + i;
                    const int slot = run.first_slot + i;
                    
                    // Calculate file offset for this compressed expert (taking into account the 4096-byte padding of the lookup table and chunks)
                    uint64_t comp_offset = info.file_offset + ((sizeof(uint32_t) * info.n_experts + 4095) & ~4095ULL);
                    for (int j = 0; j < eid; ++j) {
                        comp_offset += (info.expert_sizes_comp[j] + 4095) & ~4095ULL;
                    }
                    const uint64_t comp_size = info.expert_sizes_comp[eid];
                    const uint64_t dest = pool.cuda_ptr + uint64_t(slots_[slot].pool_index) * pool.slot_size;
                    
                    LLAMA_DSTORAGE_DEBUG_LOG(
                            "DStorage DEBUG slots:batch enqueue DECOMPRESS layer=%d expert=%d slot=%d tensor=%s comp_bytes=%" PRIu64 " uncomp_bytes=%" PRIu64 " file_off=%" PRIu64 " dest=0x%" PRIx64 "\n",
                            layer_idx, eid, slot, tname.c_str(), comp_size, info.expert_stride, comp_offset, dest);
                    
                    DSLoaderStreamRequest req;
                    req.file_path = info.file_path.c_str();
                    req.file_offset = comp_offset;
                    req.size = comp_size;
                    req.cuda_dest_ptr = dest;
                    req.uncompressed_size = info.expert_stride;
                    batch_requests.push_back(req);
                }
                continue;
            }

            if (run_experts > 1 && (pool.slot_size != info.expert_stride || info.sidecar_expert_major)) {
                for (int i = 0; i < run_experts; ++i) {
                    const int32_t eid = run.first_eid + i;
                    const int slot = run.first_slot + i;
                    const uint64_t file_off = llama_dstorage_expert_file_offset(info, eid);
                    const uint64_t dest = pool.cuda_ptr + uint64_t(slots_[slot].pool_index) * pool.slot_size;
                    LLAMA_DSTORAGE_DEBUG_LOG(
                            "DStorage DEBUG slots:batch enqueue layer=%d experts=%d..%d slots=%d..%d tensor=%s bytes=%" PRIu64 " file_off=%" PRIu64 " dest=0x%" PRIx64 "\n",
                            layer_idx, eid, eid, slot, slot, tname.c_str(), info.expert_stride, file_off, dest);
                    
                    DSLoaderStreamRequest req;
                    req.file_path = info.file_path.c_str();
                    req.file_offset = file_off;
                    req.size = info.expert_stride;
                    req.cuda_dest_ptr = dest;
                    req.uncompressed_size = info.expert_stride;
                    batch_requests.push_back(req);
                }
                continue;
            }

            const uint64_t file_off = llama_dstorage_expert_file_offset(info, run.first_eid);
            const uint64_t dest = pool.cuda_ptr + uint64_t(slots_[run.first_slot].pool_index) * pool.slot_size;
            const uint64_t size = uint64_t(run_experts) * info.expert_stride;
            LLAMA_DSTORAGE_DEBUG_LOG(
                    "DStorage DEBUG slots:batch enqueue layer=%d experts=%d..%d slots=%d..%d tensor=%s bytes=%" PRIu64 " file_off=%" PRIu64 " dest=0x%" PRIx64 "\n",
                    layer_idx, run.first_eid, run.last_eid, run.first_slot, run.last_slot,
                    tname.c_str(), size, file_off, dest);
            
            DSLoaderStreamRequest req;
            req.file_path = info.file_path.c_str();
            req.file_offset = file_off;
            req.size = size;
            req.cuda_dest_ptr = dest;
            req.uncompressed_size = size;
            batch_requests.push_back(req);
        }
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "build_batch_requests", t0, llama_dstorage_now_us());
#endif

#if LLAMA_DSTORAGE_HAS_BACKEND
    if (!batch_requests.empty()) {
        LLAMA_DSTORAGE_DEBUG_LOG("DStorage DEBUG slots:batch submit layer=%d experts=%zu requests=%zu\n",
                layer_idx, load_plan.size(), batch_requests.size());

        t0 = llama_dstorage_now_us();
        if (llama_dstorage_sort_batch_by_offset()) {
            std::stable_sort(
                    batch_requests.begin(),
                    batch_requests.end(),
                    [](const DSLoaderStreamRequest & a, const DSLoaderStreamRequest & b) {
                        return a.file_offset < b.file_offset;
                    });
        }
        uint64_t stream_file_bytes = 0;
        uint64_t stream_cuda_bytes = 0;
        uint64_t aligned_4k_requests = 0;
        uint64_t unaligned_4k_requests = 0;
        uint64_t file_offset_4k_aligned = 0;
        uint64_t size_4k_aligned = 0;
        uint64_t cuda_dest_4k_aligned = 0;
        uint64_t uncompressed_4k_aligned = 0;
        uint64_t file_offset_mod_min = UINT64_MAX;
        uint64_t file_offset_mod_max = 0;
        uint64_t file_offset_mod_gcd = 0;
        uint64_t size_mod_min = UINT64_MAX;
        uint64_t size_mod_max = 0;
        uint64_t size_mod_gcd = 0;
        uint64_t cuda_dest_mod_min = UINT64_MAX;
        uint64_t cuda_dest_mod_max = 0;
        uint64_t cuda_dest_mod_gcd = 0;
        uint64_t uncompressed_mod_min = UINT64_MAX;
        uint64_t uncompressed_mod_max = 0;
        uint64_t uncompressed_mod_gcd = 0;
        uint64_t contiguous_request_pairs = 0;
        uint64_t forward_gap_bytes = 0;
        uint64_t backward_or_overlap_pairs = 0;
        uint64_t size_le_64k = 0;
        uint64_t size_le_256k = 0;
        uint64_t size_le_1m = 0;
        uint64_t size_le_4m = 0;
        uint64_t size_gt_4m = 0;
        uint64_t min_request_bytes = UINT64_MAX;
        uint64_t max_request_bytes = 0;
        bool have_previous_request = false;
        uint64_t previous_request_end = 0;
        for (const DSLoaderStreamRequest & req : batch_requests) {
            stream_file_bytes += req.size;
            stream_cuda_bytes += req.uncompressed_size;
            min_request_bytes = std::min(min_request_bytes, req.size);
            max_request_bytes = std::max(max_request_bytes, req.size);
            if ((req.file_offset % 4096) == 0 && (req.size % 4096) == 0) {
                aligned_4k_requests++;
            } else {
                unaligned_4k_requests++;
            }
            const uint64_t file_offset_mod = req.file_offset % 4096;
            const uint64_t size_mod = req.size % 4096;
            const uint64_t cuda_dest_mod = req.cuda_dest_ptr % 4096;
            const uint64_t uncompressed_mod = req.uncompressed_size % 4096;
            file_offset_4k_aligned += file_offset_mod == 0 ? 1 : 0;
            size_4k_aligned += size_mod == 0 ? 1 : 0;
            cuda_dest_4k_aligned += cuda_dest_mod == 0 ? 1 : 0;
            uncompressed_4k_aligned += uncompressed_mod == 0 ? 1 : 0;
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
            if (uncompressed_mod != 0) {
                uncompressed_mod_min = std::min(uncompressed_mod_min, uncompressed_mod);
                uncompressed_mod_max = std::max(uncompressed_mod_max, uncompressed_mod);
                uncompressed_mod_gcd = uncompressed_mod_gcd == 0
                        ? uncompressed_mod
                        : std::gcd(uncompressed_mod_gcd, uncompressed_mod);
            }
            if (req.size <= 64ULL * 1024ULL) {
                size_le_64k++;
            } else if (req.size <= 256ULL * 1024ULL) {
                size_le_256k++;
            } else if (req.size <= 1024ULL * 1024ULL) {
                size_le_1m++;
            } else if (req.size <= 4ULL * 1024ULL * 1024ULL) {
                size_le_4m++;
            } else {
                size_gt_4m++;
            }
            if (have_previous_request) {
                if (req.file_offset == previous_request_end) {
                    contiguous_request_pairs++;
                } else if (req.file_offset > previous_request_end) {
                    forward_gap_bytes += req.file_offset - previous_request_end;
                } else {
                    backward_or_overlap_pairs++;
                }
            }
            previous_request_end = req.file_offset + req.size;
            have_previous_request = true;
        }
        if (min_request_bytes == UINT64_MAX) {
            min_request_bytes = 0;
        }
        const uint64_t max_staging_bytes = llama_dstorage_max_batch_staging_bytes();
        int result = 0;
        size_t chunk_begin = 0;
        uint64_t stream_chunks = 0;
        uint64_t stream_staging_bytes = 0;
        uint64_t max_chunk_staging_bytes = 0;
        uint64_t stream_submit_us = 0;
        uint64_t stream_wait_us = 0;
        bool bundle_contiguous_reads = llama_dstorage_bundle_contiguous_reads();
        if (bundle_contiguous_reads) {
            for (const DSLoaderStreamRequest & req : batch_requests) {
                if ((req.file_offset % 4096) != 0 || (req.size % 4096) != 0) {
                    bundle_contiguous_reads = false;
                    break;
                }
            }
        }
        const uint64_t bundle_max_read_bytes =
                bundle_contiguous_reads ? llama_dstorage_bundle_max_read_bytes() : 0;
        const uint64_t bundle_stripe_bytes =
                bundle_contiguous_reads ? llama_dstorage_bundle_stripe_bytes() : 0;
        if (bundle_contiguous_reads) {
            if (bundle_staging_size_ < max_staging_bytes) {
                if (bundle_staging_ptr_ != 0) {
                    ds_loader_cuda_free(bundle_staging_ptr_);
                    bundle_staging_ptr_ = 0;
                    bundle_staging_size_ = 0;
                }
                bundle_staging_ptr_ = ds_loader_cuda_alloc(max_staging_bytes);
                if (bundle_staging_ptr_ != 0) {
                    bundle_staging_size_ = max_staging_bytes;
                }
            }
            if (bundle_staging_ptr_ == 0) {
                LLAMA_LOG_WARN(
                        "%s: failed to allocate %.2f MiB CUDA staging buffer for bundled reads; falling back to direct reads\n",
                        __func__, max_staging_bytes / 1024.0 / 1024.0);
                bundle_contiguous_reads = false;
            }
        }
        while (chunk_begin < batch_requests.size()) {
            size_t chunk_end = chunk_begin;
            uint64_t chunk_staging_bytes = 0;
            while (chunk_end < batch_requests.size()) {
                const DSLoaderStreamRequest & req = batch_requests[chunk_end];
                const bool is_compressed = req.uncompressed_size > req.size;
                const uint64_t align_offset = is_compressed ? 0 : req.file_offset % 4096;
                const uint64_t request_staging_bytes =
                        ((chunk_staging_bytes + 4095) & ~4095ULL) +
                        (is_compressed ? req.uncompressed_size : req.size + align_offset);
                if (chunk_end > chunk_begin && request_staging_bytes > max_staging_bytes) {
                    break;
                }
                chunk_staging_bytes = request_staging_bytes;
                ++chunk_end;
            }

            LLAMA_DSTORAGE_DEBUG_LOG(
                    "DStorage DEBUG slots:batch chunk layer=%d requests=%zu..%zu staging_mib=%.2f\n",
                    layer_idx, chunk_begin, chunk_end,
                    chunk_staging_bytes / 1024.0 / 1024.0);
            const int64_t submit_t0 = llama_dstorage_now_us();
            int wait_result = 0;
            if (bundle_contiguous_reads) {
                struct BundleCopy {
                    uint64_t dst = 0;
                    uint64_t src = 0;
                    uint64_t size = 0;
                };

                std::vector<DSLoaderStreamRequest> bundled_requests;
                std::vector<BundleCopy> bundled_copies;
                bundled_requests.reserve(chunk_end - chunk_begin);
                bundled_copies.reserve(chunk_end - chunk_begin);

                uint64_t staging_cursor = 0;
                size_t i = chunk_begin;
                while (i < chunk_end) {
                    const DSLoaderStreamRequest & first = batch_requests[i];
                    const size_t group_start = i;
                    uint64_t group_end_offset = first.file_offset + first.size;
                    ++i;
                    while (i < chunk_end) {
                        const DSLoaderStreamRequest & next = batch_requests[i];
                        const bool same_file =
                                first.file_path == next.file_path ||
                                (first.file_path != nullptr && next.file_path != nullptr &&
                                 std::wcscmp(first.file_path, next.file_path) == 0);
                        if (!same_file || next.file_offset != group_end_offset) {
                            break;
                        }
                        if (bundle_max_read_bytes > 0 &&
                                (next.file_offset + next.size - first.file_offset) > bundle_max_read_bytes) {
                            break;
                        }
                        group_end_offset += next.size;
                        ++i;
                    }

                    staging_cursor = (staging_cursor + 4095ULL) & ~4095ULL;
                    const uint64_t group_staging_offset = staging_cursor;
                    const uint64_t group_size = group_end_offset - first.file_offset;
                    if (group_staging_offset + group_size > max_staging_bytes) {
                        result = -1;
                        break;
                    }

                    uint64_t stripe_offset = 0;
                    while (stripe_offset < group_size) {
                        const uint64_t stripe_size = bundle_stripe_bytes > 0 ?
                                std::min(bundle_stripe_bytes, group_size - stripe_offset) :
                                group_size - stripe_offset;
                        DSLoaderStreamRequest bundle_req = {};
                        bundle_req.file_path = first.file_path;
                        bundle_req.file_offset = first.file_offset + stripe_offset;
                        bundle_req.size = stripe_size;
                        bundle_req.cuda_dest_ptr = bundle_staging_ptr_ + group_staging_offset + stripe_offset;
                        bundle_req.uncompressed_size = stripe_size;
                        bundled_requests.push_back(bundle_req);
                        stripe_offset += stripe_size;
                    }

                    for (size_t j = group_start; j < i; ++j) {
                        const DSLoaderStreamRequest & original = batch_requests[j];
                        bundled_copies.push_back({
                                original.cuda_dest_ptr,
                                bundle_staging_ptr_ + group_staging_offset + (original.file_offset - first.file_offset),
                                original.uncompressed_size });
                    }
                    staging_cursor = group_staging_offset + group_size;
                }

                if (result == 0) {
                    result = ds_loader_stream_to_cuda_batch(
                            ds_loader_,
                            bundled_requests.data(),
                            int(bundled_requests.size()));
                }
                const int64_t submit_t1_inner = llama_dstorage_now_us();
                const int64_t wait_t0_inner = llama_dstorage_now_us();
                wait_result = result == 0 ? ds_loader_cuda_wait_event(ds_loader_) : -1;
                if (wait_result == 0) {
                    for (const BundleCopy & copy : bundled_copies) {
                        if (ds_loader_cuda_dtod(copy.dst, copy.src, copy.size) != 0) {
                            wait_result = -1;
                            break;
                        }
                    }
                }
                const int64_t wait_t1_inner = llama_dstorage_now_us();
                stream_submit_us += uint64_t(std::max<int64_t>(0, submit_t1_inner - submit_t0));
                stream_wait_us += uint64_t(std::max<int64_t>(0, wait_t1_inner - wait_t0_inner));
                llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "cuda_copy_wait", wait_t0_inner, wait_t1_inner);
            } else {
                result = ds_loader_stream_to_cuda_batch(
                        ds_loader_,
                        batch_requests.data() + chunk_begin,
                        int(chunk_end - chunk_begin));
                const int64_t submit_t1_inner = llama_dstorage_now_us();
                const int64_t wait_t0_inner = llama_dstorage_now_us();
                wait_result = result == 0 ? ds_loader_cuda_wait_event(ds_loader_) : -1;
                const int64_t wait_t1_inner = llama_dstorage_now_us();
                stream_submit_us += uint64_t(std::max<int64_t>(0, submit_t1_inner - submit_t0));
                stream_wait_us += uint64_t(std::max<int64_t>(0, wait_t1_inner - wait_t0_inner));
                llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "cuda_copy_wait", wait_t0_inner, wait_t1_inner);
            }
            stream_staging_bytes += chunk_staging_bytes;
            max_chunk_staging_bytes = std::max(max_chunk_staging_bytes, chunk_staging_bytes);
            stream_chunks++;
            if (result != 0 || wait_result != 0) {
                result = -1;
                break;
            }
            chunk_begin = chunk_end;
        }
        llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "ds_loader_stream_to_cuda_batch", t0, llama_dstorage_now_us());

        if (result != 0) {
            LLAMA_DSTORAGE_DEBUG_LOG(
                    "DStorage DEBUG slots:batch failed layer=%d result=%d hr=0x%08" PRIx32 "\n",
                    layer_idx, result, uint32_t(ds_loader_get_hresult()));
            LLAMA_LOG_ERROR("DirectStorage: failed to batch stream L%d (%zu experts, %zu requests), hr=0x%08" PRIx32 "\n",
                    layer_idx, load_plan.size(), batch_requests.size(), uint32_t(ds_loader_get_hresult()));
            return false;
        }

        const double transfer_us = double(llama_dstorage_now_us() - t0);
        trace_stream_file_bytes = stream_file_bytes;
        trace_transfer_us = uint64_t(std::max<double>(0.0, transfer_us));
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            transfer_ewma_us_ = transfer_ewma_us_ == 0.0
                    ? transfer_us
                    : transfer_ewma_us_ * 0.8 + transfer_us * 0.2;
        }
        phase_stats.stream_calls++;
        phase_stats.stream_cold_experts += uint64_t(cold_to_load.size());
        phase_stats.stream_pinned_experts += uint64_t(pinned_to_load.size());
        phase_stats.stream_possible_runs += possible_run_count;
        phase_stats.stream_possible_run_experts += possible_run_expert_count;
        phase_stats.stream_runs += uint64_t(runs.size());
        phase_stats.stream_run_experts += run_expert_count;
        phase_stats.stream_requests += uint64_t(batch_requests.size());
        phase_stats.stream_chunks += stream_chunks;
        phase_stats.stream_file_bytes += stream_file_bytes;
        phase_stats.stream_cuda_bytes += stream_cuda_bytes;
        phase_stats.stream_staging_bytes += stream_staging_bytes;
        phase_stats.stream_submit_us += stream_submit_us;
        phase_stats.stream_wait_us += stream_wait_us;
        phase_stats.stream_total_us += uint64_t(std::max<int64_t>(0, int64_t(transfer_us)));
        phase_stats.stream_max_requests_per_call = std::max(
                phase_stats.stream_max_requests_per_call,
                uint64_t(batch_requests.size()));
        phase_stats.stream_max_chunks_per_call = std::max(
                phase_stats.stream_max_chunks_per_call,
                stream_chunks);
        phase_stats.stream_aligned_4k_requests += aligned_4k_requests;
        phase_stats.stream_unaligned_4k_requests += unaligned_4k_requests;
        phase_stats.stream_file_offset_4k_aligned += file_offset_4k_aligned;
        phase_stats.stream_size_4k_aligned += size_4k_aligned;
        phase_stats.stream_cuda_dest_4k_aligned += cuda_dest_4k_aligned;
        phase_stats.stream_uncompressed_4k_aligned += uncompressed_4k_aligned;
        if (file_offset_mod_min != UINT64_MAX) {
            phase_stats.stream_file_offset_mod_min = std::min(
                    phase_stats.stream_file_offset_mod_min,
                    file_offset_mod_min);
            phase_stats.stream_file_offset_mod_max = std::max(
                    phase_stats.stream_file_offset_mod_max,
                    file_offset_mod_max);
            phase_stats.stream_file_offset_mod_gcd = phase_stats.stream_file_offset_mod_gcd == 0
                    ? file_offset_mod_gcd
                    : std::gcd(phase_stats.stream_file_offset_mod_gcd, file_offset_mod_gcd);
        }
        if (size_mod_min != UINT64_MAX) {
            phase_stats.stream_size_mod_min = std::min(
                    phase_stats.stream_size_mod_min,
                    size_mod_min);
            phase_stats.stream_size_mod_max = std::max(
                    phase_stats.stream_size_mod_max,
                    size_mod_max);
            phase_stats.stream_size_mod_gcd = phase_stats.stream_size_mod_gcd == 0
                    ? size_mod_gcd
                    : std::gcd(phase_stats.stream_size_mod_gcd, size_mod_gcd);
        }
        if (cuda_dest_mod_min != UINT64_MAX) {
            phase_stats.stream_cuda_dest_mod_min = std::min(
                    phase_stats.stream_cuda_dest_mod_min,
                    cuda_dest_mod_min);
            phase_stats.stream_cuda_dest_mod_max = std::max(
                    phase_stats.stream_cuda_dest_mod_max,
                    cuda_dest_mod_max);
            phase_stats.stream_cuda_dest_mod_gcd = phase_stats.stream_cuda_dest_mod_gcd == 0
                    ? cuda_dest_mod_gcd
                    : std::gcd(phase_stats.stream_cuda_dest_mod_gcd, cuda_dest_mod_gcd);
        }
        if (uncompressed_mod_min != UINT64_MAX) {
            phase_stats.stream_uncompressed_mod_min = std::min(
                    phase_stats.stream_uncompressed_mod_min,
                    uncompressed_mod_min);
            phase_stats.stream_uncompressed_mod_max = std::max(
                    phase_stats.stream_uncompressed_mod_max,
                    uncompressed_mod_max);
            phase_stats.stream_uncompressed_mod_gcd = phase_stats.stream_uncompressed_mod_gcd == 0
                    ? uncompressed_mod_gcd
                    : std::gcd(phase_stats.stream_uncompressed_mod_gcd, uncompressed_mod_gcd);
        }
        phase_stats.stream_contiguous_request_pairs += contiguous_request_pairs;
        phase_stats.stream_forward_gap_bytes += forward_gap_bytes;
        phase_stats.stream_backward_or_overlap_pairs += backward_or_overlap_pairs;
        phase_stats.stream_size_le_64k += size_le_64k;
        phase_stats.stream_size_le_256k += size_le_256k;
        phase_stats.stream_size_le_1m += size_le_1m;
        phase_stats.stream_size_le_4m += size_le_4m;
        phase_stats.stream_size_gt_4m += size_gt_4m;
        phase_stats.stream_min_request_bytes = std::min(
                phase_stats.stream_min_request_bytes,
                min_request_bytes);
        phase_stats.stream_max_request_bytes = std::max(
                phase_stats.stream_max_request_bytes,
                max_request_bytes);
        phase_stats.stream_max_chunk_staging_bytes = std::max(
                phase_stats.stream_max_chunk_staging_bytes,
                max_chunk_staging_bytes);
    }
#endif

#if LLAMA_DSTORAGE_HAS_BACKEND
    if (pinned_cache_enabled() && pinned_decode_like) {
        t0 = llama_dstorage_now_us();
        uint32_t queued = 0;
        uint32_t skipped = 0;
        const uint32_t admit_min_hits = pinned_admit_min_hits();
        const uint32_t admit_max_per_call = llama_dstorage_pinned_admit_max_per_call();
        const uint64_t admit_max_pending_bytes = llama_dstorage_pinned_admit_max_pending_bytes();
        std::vector<PinnedAdmissionWork> admission_batch;

        struct PinnedAdmissionCandidate {
            uint64_t frequency = 0;
            uint64_t entry_bytes = 0;
            PinnedAdmissionWork work;
        };
        std::vector<PinnedAdmissionCandidate> candidates;

        for (const LoadPlan & plan : load_plan) {
            if (plan.from_pinned) {
                continue;
            }

            const uint64_t expert_key = make_expert_key(layer_idx, plan.eid);
            if (pinned_cache_.find(expert_key) != pinned_cache_.end()) {
                continue;
            }
            if (pinned_admission_pending(expert_key)) {
                skipped++;
                diagnostic_stats_.pinned_admit_skipped_pending++;
                continue;
            }
            const uint64_t frequency = pinned_frequency_[expert_key];
            if (frequency < admit_min_hits) {
                skipped++;
                diagnostic_stats_.pinned_admit_skipped_policy++;
                continue;
            }

            uint64_t entry_bytes = 0;
            std::vector<PinnedSlice> slices;
            std::vector<PinnedReadOp> read_ops;
            slices.reserve(layer_tnames.size());
            read_ops.reserve(layer_tnames.size());
            bool can_admit = true;
            for (const std::string & tname : layer_tnames) {
                const std::string tkey = make_tensor_key(layer_idx, tname.c_str());
                auto reg_it = tensor_registry_.find(tkey);
                if (reg_it == tensor_registry_.end()) {
                    can_admit = false;
                    break;
                }
                const ExpertTensorInfo & info = reg_it->second;
                if (info.total_size_compressed > 0 || info.expert_stride == 0 || plan.eid >= info.n_experts) {
                    can_admit = false;
                    break;
                }
                slices.push_back({ tname, entry_bytes, info.expert_stride });
                read_ops.push_back({
                        info.file_path,
                        llama_dstorage_expert_file_offset(info, plan.eid),
                        entry_bytes,
                        info.expert_stride });
                entry_bytes += info.expert_stride;
            }

            if (!can_admit || entry_bytes == 0 || entry_bytes > pinned_budget_bytes_) {
                skipped++;
                diagnostic_stats_.pinned_admit_skipped_policy++;
                continue;
            }

            PinnedEntry entry;
            entry.layer_idx = layer_idx;
            entry.expert_idx = plan.eid;
            entry.host_ptr = nullptr;
            entry.bytes = entry_bytes;
            entry.is_pinned = false;
            entry.last_used_tick = 0;
            entry.slices = std::move(slices);

            PinnedAdmissionWork work;
            work.result.expert_key = expert_key;
            work.result.slot_idx = plan.target_slot;
            work.result.ok = false;
            work.result.entry = std::move(entry);
            work.read_ops = std::move(read_ops);

            candidates.push_back({ frequency, entry_bytes, std::move(work) });
        }

        std::sort(
                candidates.begin(),
                candidates.end(),
                [](const PinnedAdmissionCandidate & a, const PinnedAdmissionCandidate & b) {
                    if (a.frequency != b.frequency) {
                        return a.frequency > b.frequency;
                    }
                    return a.entry_bytes < b.entry_bytes;
                });

        uint64_t admission_batch_bytes = 0;
        for (PinnedAdmissionCandidate & candidate : candidates) {
            if (admit_max_per_call == 0 || queued >= admit_max_per_call) {
                skipped++;
                diagnostic_stats_.pinned_admit_skipped_throttle++;
                continue;
            }
            if (admit_max_pending_bytes > 0 &&
                    pinned_pending_bytes_ + admission_batch_bytes + candidate.entry_bytes > admit_max_pending_bytes) {
                skipped++;
                diagnostic_stats_.pinned_admit_skipped_throttle++;
                continue;
            }

            while (pinned_used_bytes_ + candidate.entry_bytes > pinned_budget_bytes_ && !pinned_cache_.empty()) {
                auto victim = pinned_cache_.end();
                uint32_t victim_freq = std::numeric_limits<uint32_t>::max();
                uint64_t victim_tick = std::numeric_limits<uint64_t>::max();
                for (auto it = pinned_cache_.begin(); it != pinned_cache_.end(); ++it) {
                    if (static_pinned_entry(it->first)) {
                        continue;
                    }
                    const uint32_t freq = pinned_frequency_[it->first];
                    const uint64_t tick = it->second.last_used_tick;
                    if (freq < victim_freq || (freq == victim_freq && tick < victim_tick)) {
                        victim = it;
                        victim_freq = freq;
                        victim_tick = tick;
                    }
                }
                if (victim == pinned_cache_.end()) {
                    break;
                }
                LLAMA_DSTORAGE_DEBUG_LOG(
                        "DStorage DEBUG pinned:evict old=(L%d,E%d) bytes_mib=%.4f freq=%u\n",
                        victim->second.layer_idx, victim->second.expert_idx,
                        victim->second.bytes / 1024.0 / 1024.0, pinned_frequency_[victim->first]);
                if (victim->second.host_ptr != nullptr) {
                    ds_loader_host_free(victim->second.host_ptr, victim->second.is_pinned ? 1 : 0);
                }
                pinned_used_bytes_ -= victim->second.bytes;
                pinned_cache_.erase(victim);
            }

            if (pinned_used_bytes_ + candidate.entry_bytes > pinned_budget_bytes_) {
                skipped++;
                diagnostic_stats_.pinned_admit_skipped_budget++;
                continue;
            }

            pinned_used_bytes_ += candidate.entry_bytes;
            admission_batch_bytes += candidate.entry_bytes;
            slots_[candidate.work.result.slot_idx].admission_pending = true;
            admission_batch.push_back(std::move(candidate.work));
            queued++;
        }

        if (!admission_batch.empty()) {
            std::vector<uint64_t> pending_keys;
            std::vector<int> pending_slots;
            pending_keys.reserve(admission_batch.size());
            pending_slots.reserve(admission_batch.size());
            for (const PinnedAdmissionWork & work : admission_batch) {
                pending_keys.push_back(work.result.expert_key);
                pending_slots.push_back(work.result.slot_idx);
            }

            std::shared_ptr<std::vector<PinnedAdmissionWork>> batch_ptr;
            try {
                batch_ptr = std::make_shared<std::vector<PinnedAdmissionWork>>(std::move(admission_batch));
                std::future<std::vector<PinnedAdmissionResult>> future = std::async(
                        std::launch::async,
                        [batch_ptr]() mutable {
                            std::vector<PinnedAdmissionResult> results;
                            results.reserve(batch_ptr->size());
                            for (PinnedAdmissionWork & work : *batch_ptr) {
                                const int64_t admit_t0 = llama_dstorage_now_us();
                                int is_pinned = 0;
                                work.result.entry.host_ptr = ds_loader_host_alloc(work.result.entry.bytes, &is_pinned);
                                work.result.entry.is_pinned = is_pinned != 0;
                                if (work.result.entry.host_ptr == nullptr) {
                                    work.result.ok = false;
                                    work.result.admit_us = uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - admit_t0));
                                    results.push_back(std::move(work.result));
                                    continue;
                                }

                                bool ok = true;
                                std::unordered_map<std::string, int> open_files;
                                for (const PinnedReadOp & op : work.read_ops) {
                                    const std::string path(op.file_path.begin(), op.file_path.end());
                                    auto fd_it = open_files.find(path);
                                    if (fd_it == open_files.end()) {
                                        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
                                        if (fd < 0) {
                                            ok = false;
                                            break;
                                        }
                                        fd_it = open_files.emplace(path, fd).first;
                                    }
                                    char * dst = reinterpret_cast<char *>(work.result.entry.host_ptr) + op.dst_offset;
                                    if (!llama_dstorage_pread_full(fd_it->second, op.file_offset, dst, op.size)) {
                                        ok = false;
                                        break;
                                    }
                                }
                                for (const auto & kv : open_files) {
                                    close(kv.second);
                                }
                                work.result.ok = ok;
                                work.result.admit_us = uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - admit_t0));
                                results.push_back(std::move(work.result));
                            }
                            return results;
                        });
                pinned_admissions_.push_back({ std::move(pending_keys), std::move(pending_slots), admission_batch_bytes, std::move(future) });
                pinned_pending_bytes_ += admission_batch_bytes;
                diagnostic_stats_.pinned_admit_queued += queued;
            } catch (...) {
                std::vector<PinnedAdmissionWork> & failed_batch = batch_ptr ? *batch_ptr : admission_batch;
                for (PinnedAdmissionWork & work : failed_batch) {
                    if (work.result.slot_idx >= 0 && work.result.slot_idx < int(slots_.size())) {
                        slots_[work.result.slot_idx].admission_pending = false;
                    }
                    if (pinned_used_bytes_ >= work.result.entry.bytes) {
                        pinned_used_bytes_ -= work.result.entry.bytes;
                    } else {
                        pinned_used_bytes_ = 0;
                    }
                    if (work.result.entry.host_ptr != nullptr) {
                        ds_loader_host_free(work.result.entry.host_ptr, work.result.entry.is_pinned ? 1 : 0);
                    }
                }
                skipped += queued;
                queued = 0;
            }
        }

        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG pinned:admit_async layer=%d queued=%u skipped=%u pending=%zu entries=%zu used_mib=%.2f budget_mib=%.2f min_hits=%u\n",
                layer_idx, queued, skipped, pinned_admissions_.size(), pinned_cache_.size(),
                pinned_used_bytes_ / 1024.0 / 1024.0,
                pinned_budget_bytes_ / 1024.0 / 1024.0,
                admit_min_hits);
        llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "pinned_admission", t0, llama_dstorage_now_us());
    } else if (pinned_cache_enabled()) {
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG pinned:admit layer=%d skipped_prefill n_selected=%d entries=%zu used_mib=%.2f budget_mib=%.2f\n",
                layer_idx, n_selected, pinned_cache_.size(),
                pinned_used_bytes_ / 1024.0 / 1024.0,
                pinned_budget_bytes_ / 1024.0 / 1024.0);
    }
#endif

    t0 = llama_dstorage_now_us();
    for (const LoadPlan & plan : load_plan) {
        // Mark slot occupied (shared across all types)
        slots_[plan.target_slot].layer_idx  = layer_idx;
        slots_[plan.target_slot].expert_idx = plan.eid;
        slots_[plan.target_slot].occupied   = true;
        slots_[plan.target_slot].transfer_pending = false;
        slots_[plan.target_slot].last_used_tick = ++use_tick_;
        if (accounting_phase == dstorage_moe_phase::decode) {
            slots_[plan.target_slot].decode_hits = 1000;
        } else if (accounting_phase == dstorage_moe_phase::prefetch) {
            slots_[plan.target_slot].prefetch_hits = 1000;
        } else {
            slots_[plan.target_slot].prefill_hits = 1000;
        }
        arc_record_admission(slots_[plan.target_slot], accounting_phase);
        expert_to_slot[plan.eid] = plan.target_slot;
    }
    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "mark_slots_occupied", t0, llama_dstorage_now_us());

    if (phase == dstorage_moe_phase::prefetch) {
        out_id_map.clear();
        out_pool_ptrs.clear();
    } else {
        t0 = llama_dstorage_now_us();
        const int active_count = (int) unique_experts.size();
        if (phase == dstorage_moe_phase::prefill &&
                active_count > max_prefill_workspace_experts()) {
            LLAMA_LOG_ERROR(
                    "%s: prefill workspace requires %d unique experts, exceeding bounded capacity %d\n",
                    __func__,
                    active_count,
                    max_prefill_workspace_experts());
            return false;
        }
        if (phase == dstorage_moe_phase::decode &&
                calculate_decode_workspace_capacity(active_count, n_expert_used_) <= 0) {
            LLAMA_LOG_ERROR(
                    "%s: invalid decode request for %d unique experts (top-k=%d)\n",
                    __func__,
                    active_count,
                    n_expert_used_);
            return false;
        }
        active_slot_count_ = 0;

        // The persistent pools are contiguous slot arrays. Remapping expert IDs
        // to global slot indices lets MUL_MAT_ID consume cached experts directly.
        out_id_map.resize(n_selected);
        for (int i = 0; i < n_selected; i++) {
            auto it = expert_to_slot.find(expert_ids[i]);
            if (it == expert_to_slot.end()) {
                LLAMA_LOG_ERROR(
                        "%s: missing global slot for layer %d expert %d\n",
                        __func__,
                        layer_idx,
                        expert_ids[i]);
                return false;
            }
            out_id_map[i] = slots_[it->second].pool_index;
        }
        llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "build_global_slot_id_map", t0, llama_dstorage_now_us());

        t0 = llama_dstorage_now_us();
        for (const std::string & tname : layer_tnames) {
            const std::string pool_key = pool_key_for_layer(layer_idx, tname);
            auto pool_it = type_pools_.find(pool_key);
            if (pool_it == type_pools_.end()) {
                LLAMA_LOG_ERROR(
                        "%s: missing persistent pool for tensor type '%s'\n",
                        __func__,
                        pool_key.c_str());
                return false;
            }
            out_pool_ptrs[tname] = pool_it->second.cuda_ptr;
        }
        llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "build_persistent_pool_ptrs", t0, llama_dstorage_now_us());

        auto & executing_slots = execution_slots_by_layer_[layer_idx];
        for (int slot_idx : executing_slots) {
            if (slot_idx >= 0 &&
                    slot_idx < int(slots_.size()) &&
                    slots_[slot_idx].execution_pins > 0) {
                slots_[slot_idx].execution_pins--;
            }
        }
        executing_slots.clear();
        executing_slots.reserve(unique_experts.size());
        for (int32_t eid : unique_experts) {
            const auto it = expert_to_slot.find(eid);
            if (it == expert_to_slot.end()) {
                return false;
            }
            slots_[it->second].execution_pins++;
            executing_slots.push_back(it->second);
        }
    }

    const uint64_t ensure_total_us =
            uint64_t(std::max<int64_t>(0, llama_dstorage_now_us() - t_total0));
    if (phase != dstorage_moe_phase::prefetch) {
        const uint64_t hit_count = uint64_t(unique_experts.size() - to_load.size());
        const uint64_t miss_count = uint64_t(to_load.size());
        timeline_stats_.real_calls++;
        if (accounting_phase == dstorage_moe_phase::decode) {
            timeline_stats_.decode_calls++;
        } else {
            timeline_stats_.prefill_calls++;
        }
        if (miss_count == 0) {
            timeline_stats_.all_hit_calls++;
        } else {
            timeline_stats_.miss_calls++;
        }
        timeline_stats_.selected_experts += uint64_t(n_selected);
        timeline_stats_.unique_experts += uint64_t(unique_experts.size());
        timeline_stats_.hit_experts += hit_count;
        timeline_stats_.miss_experts += miss_count;
        timeline_stats_.ensure_total_us += ensure_total_us;
        timeline_stats_.stream_total_us += trace_transfer_us;
        timeline_stats_.stream_file_bytes += trace_stream_file_bytes;

        if (miss_count > 0) {
            const uint64_t prior_compute_us = last_layer_compute_us_;
            const uint64_t hidden_us = std::min(prior_compute_us, trace_transfer_us);
            const uint64_t visible_us = trace_transfer_us > hidden_us
                    ? trace_transfer_us - hidden_us
                    : 0;
            uint64_t hideable_file_bytes = 0;
            if (trace_transfer_us > 0 && trace_stream_file_bytes > 0) {
                const long double fit =
                        (long double) trace_stream_file_bytes *
                        (long double) hidden_us /
                        (long double) trace_transfer_us;
                hideable_file_bytes = uint64_t(std::min<long double>(
                            (long double) trace_stream_file_bytes,
                            fit));
            } else if (trace_transfer_us == 0) {
                hideable_file_bytes = trace_stream_file_bytes;
            }

            const int64_t margin_us = int64_t(prior_compute_us) - int64_t(trace_transfer_us);
            auto record_dry_run = [&](SchedulerDryRunStats & dry) {
                dry.miss_calls++;
                dry.transfer_total_us += trace_transfer_us;
                dry.prior_compute_total_us += prior_compute_us;
                dry.hidden_possible_us += hidden_us;
                dry.visible_after_overlap_us += visible_us;
                dry.file_bytes += trace_stream_file_bytes;
                dry.hideable_file_bytes += hideable_file_bytes;
                if (prior_compute_us == 0) {
                    dry.no_window_calls++;
                } else if (trace_transfer_us <= prior_compute_us) {
                    dry.full_hide_calls++;
                } else {
                    dry.partial_hide_calls++;
                }
                dry.best_margin_us = std::max(dry.best_margin_us, margin_us);
                dry.worst_margin_us = std::min(dry.worst_margin_us, margin_us);
            };
            record_dry_run(scheduler_dry_run_stats_);
            if (accounting_phase == dstorage_moe_phase::decode) {
                record_dry_run(scheduler_decode_dry_run_stats_);
            } else {
                record_dry_run(scheduler_prefill_dry_run_stats_);
            }

            if (llama_dstorage_timeline_trace_enabled()) {
                std::fprintf(
                        stderr,
                        "DSTORAGE_MOE_TIMELINE_EVENT layer=%d phase=%s selected=%d unique=%zu"
                        " hits=%" PRIu64 " misses=%" PRIu64
                        " ensure_us=%" PRIu64 " stream_us=%" PRIu64
                        " stream_mib=%.4f prior_compute_us=%" PRIu64
                        " dry_hidden_us=%" PRIu64 " dry_visible_us=%" PRIu64
                        " dry_fit_mib=%.4f dry_can_hide=%d\n",
                        layer_idx,
                        llama_dstorage_phase_name(accounting_phase),
                        n_selected,
                        unique_experts.size(),
                        hit_count,
                        miss_count,
                        ensure_total_us,
                        trace_transfer_us,
                        trace_stream_file_bytes / 1024.0 / 1024.0,
                        prior_compute_us,
                        hidden_us,
                        visible_us,
                        hideable_file_bytes / 1024.0 / 1024.0,
                        trace_transfer_us <= prior_compute_us ? 1 : 0);
                std::fflush(stderr);
            }
        } else if (llama_dstorage_timeline_trace_enabled()) {
            std::fprintf(
                    stderr,
                    "DSTORAGE_MOE_TIMELINE_EVENT layer=%d phase=%s selected=%d unique=%zu"
                    " hits=%" PRIu64 " misses=0 ensure_us=%" PRIu64
                    " stream_us=0 stream_mib=0.0000 prior_compute_us=%" PRIu64
                    " dry_hidden_us=0 dry_visible_us=0 dry_fit_mib=0.0000 dry_can_hide=1\n",
                    layer_idx,
                    llama_dstorage_phase_name(accounting_phase),
                    n_selected,
                    unique_experts.size(),
                    hit_count,
                    ensure_total_us,
                    last_layer_compute_us_);
            std::fflush(stderr);
        }
    }

    write_routing_trace(
            layer_idx,
            phase,
            unique_experts,
            to_load,
            trace_stream_file_bytes,
            trace_transfer_us,
            ensure_total_us);

    // --- Log streaming summary ---
    if (!to_load.empty() && llama_dstorage_debug_enabled()) {
        LLAMA_LOG_INFO("DirectStorage: L%d streamed %d experts [", layer_idx, (int)to_load.size());
        for (size_t i = 0; i < to_load.size(); i++) {
            LLAMA_LOG_INFO("%d%s", to_load[i], i + 1 < to_load.size() ? "," : "");
        }
        LLAMA_LOG_INFO("] (%d cached)\n", (int)(unique_experts.size() - to_load.size()));
    }
    if (llama_dstorage_debug_enabled()) {
        uint32_t prefill_only_slots = 0;
        uint32_t decode_touched_slots = 0;
        for (const ExpertSlot & slot : slots_) {
            if (!slot.occupied) {
                continue;
            }
            if (slot.decode_hits > 0) {
                decode_touched_slots++;
            } else {
                prefill_only_slots++;
            }
        }
        LLAMA_DSTORAGE_DEBUG_LOG(
                "DStorage DEBUG slots:phase_occupancy policy=%d phase=%s prefill_only=%u decode_touched=%u total_slots=%d\n",
                phase_cache_policy_ ? 1 : 0, llama_dstorage_phase_name(phase),
                prefill_only_slots, decode_touched_slots, n_slots_);
    }

    llama_dstorage_trace_us("slots.ensure_loaded", layer_idx, "total", t_total0, llama_dstorage_now_us());
    return true;
}

void DStorageSlotManager::release_layer_experts(int layer_idx) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    const auto it = execution_slots_by_layer_.find(layer_idx);
    if (it == execution_slots_by_layer_.end()) {
        return;
    }
    for (int slot_idx : it->second) {
        if (slot_idx >= 0 &&
                slot_idx < int(slots_.size()) &&
                slots_[slot_idx].execution_pins > 0) {
            slots_[slot_idx].execution_pins--;
        }
    }
    execution_slots_by_layer_.erase(it);
}

void DStorageSlotManager::record_layer_compute_us(int64_t compute_us) {
    if (compute_us <= 0) {
        return;
    }
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    const double sample = double(compute_us);
    layer_compute_ewma_us_ = layer_compute_ewma_us_ == 0.0
            ? sample
            : layer_compute_ewma_us_ * 0.8 + sample * 0.2;
    const uint64_t sample_us = uint64_t(compute_us);
    last_layer_compute_us_ = sample_us;
    timeline_stats_.compute_samples++;
    timeline_stats_.compute_total_us += sample_us;
    timeline_stats_.compute_min_us = std::min(timeline_stats_.compute_min_us, sample_us);
    timeline_stats_.compute_max_us = std::max(timeline_stats_.compute_max_us, sample_us);
}

double DStorageSlotManager::prefetch_confidence(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected) {
    if (expert_ids == nullptr || n_selected <= 0) {
        return 0.0;
    }
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    double total = 0.0;
    int count = 0;
    for (int i = 0; i < n_selected; ++i) {
        if (expert_ids[i] < 0) {
            continue;
        }
        total += llama_dstorage_high_confidence_prefetch_enabled()
                ? request_activation_stats_.rolling_decode_confidence(layer_idx, expert_ids[i])
                : request_activation_stats_.decode_confidence(layer_idx, expert_ids[i]);
        count++;
    }
    return count > 0 ? total / count : 0.0;
}

std::vector<int32_t> DStorageSlotManager::history_prefetch_candidates(
        int layer_idx,
        int max_experts) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    const int capped_max_experts =
            llama_dstorage_high_confidence_prefetch_enabled()
                    ? llama_dstorage_prefetch_max_experts_per_layer(max_experts)
                    : max_experts;
    std::vector<int32_t> candidates =
            llama_dstorage_high_confidence_prefetch_enabled()
                    ? request_activation_stats_.top_rolling_decode_experts(
                            layer_idx,
                            capped_max_experts,
                            llama_dstorage_prefetch_min_confidence(),
                            uint32_t(llama_dstorage_prefetch_min_observations()))
                    : request_activation_stats_.top_decode_experts(
                            layer_idx,
                            capped_max_experts,
                            llama_dstorage_prefetch_min_confidence());
    candidates.erase(
            std::remove_if(
                    candidates.begin(),
                    candidates.end(),
                    [&](int32_t eid) {
                        for (const ExpertSlot & slot : slots_) {
                            if (slot.occupied &&
                                    slot.layer_idx == layer_idx &&
                                    slot.expert_idx == eid) {
                                return true;
                            }
                        }
                        return false;
                    }),
            candidates.end());
    return candidates;
}

std::vector<int32_t> DStorageSlotManager::completion_prefetch_candidates(
        int layer_idx,
        int max_experts) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    const int capped_max_experts =
            llama_dstorage_high_confidence_prefetch_enabled()
                    ? llama_dstorage_prefetch_max_experts_per_layer(max_experts)
                    : max_experts;
    std::vector<int32_t> predicted =
            llama_dstorage_high_confidence_prefetch_enabled()
                    ? request_activation_stats_.top_rolling_decode_experts(
                            layer_idx,
                            capped_max_experts,
                            llama_dstorage_prefetch_min_confidence(),
                            uint32_t(llama_dstorage_prefetch_min_observations()))
                    : request_activation_stats_.top_decode_experts(
                            layer_idx,
                            capped_max_experts,
                            llama_dstorage_prefetch_min_confidence());
    if (int(predicted.size()) < capped_max_experts) {
        return {};
    }

    std::vector<int32_t> missing;
    missing.reserve(predicted.size());
    for (int32_t eid : predicted) {
        bool resident = false;
        for (const ExpertSlot & slot : slots_) {
            if (slot.occupied &&
                    slot.layer_idx == layer_idx &&
                    slot.expert_idx == eid) {
                resident = true;
                break;
            }
        }
        if (!resident) {
            missing.push_back(eid);
        }
    }
    return missing;
}

int DStorageSlotManager::recommended_prefetch_distance(double confidence) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    int occupied = 0;
    for (const ExpertSlot & slot : slots_) {
        occupied += slot.occupied ? 1 : 0;
    }
    const double cache_pressure = n_slots_ > 0 ? double(occupied) / n_slots_ : 1.0;
    const uint64_t accesses = decode_stats_.hits + decode_stats_.misses;
    const double miss_rate = accesses > 0 ? double(decode_stats_.misses) / accesses : 1.0;
    double transfer_us, compute_us;
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        transfer_us = transfer_ewma_us_ > 0.0 ? transfer_ewma_us_ : 8000.0;
        compute_us = layer_compute_ewma_us_ > 0.0 ? layer_compute_ewma_us_ : 4000.0;
    }
    return select_prefetch_distance(
            transfer_us,
            compute_us,
            confidence,
            cache_pressure,
            miss_rate,
            llama_dstorage_max_prefetch_distance());
}

// ---------------------------------------------------------------------------
// destroy / destructor
// ---------------------------------------------------------------------------

void DStorageSlotManager::destroy() {
    collect_async_prefetches(true);
    {
        std::lock_guard<std::mutex> lock(async_prefetch_mutex_);
        global_prefetch_queue_.clear();
    }
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    if (routing_trace_file_ != nullptr) {
        std::fclose(routing_trace_file_);
        routing_trace_file_ = nullptr;
    }
    if (llama_dstorage_summary_enabled()) {
        const int speculative_slots = speculative_slot_count();
        uint64_t occupied = 0;
        uint64_t decode_touched = 0;
        uint64_t prefetch_touched = 0;
        uint64_t prefetch_only = 0;
        uint64_t prefetch_resident_unused = 0;
        uint64_t prefetch_resident_used = 0;
        uint64_t prefill_only = 0;
        uint64_t decode_hot_resident_protected = 0;
        uint64_t arc_recent = 0;
        uint64_t arc_frequent = 0;
        for (const ExpertSlot & slot : slots_) {
            if (!slot.occupied) {
                continue;
            }
            occupied++;
            if (slot.decode_hits > 0) {
                decode_touched++;
            }
            if (slot.prefetch_hits > 0) {
                prefetch_touched++;
            }
            if (slot.prefetch_hits > 0 && slot.decode_hits == 0) {
                prefetch_only++;
            }
            if (slot.prefetch_hits > 0) {
                if (slot.decode_hits == 0 && slot.prefill_hits == 0) {
                    prefetch_resident_unused++;
                } else {
                    prefetch_resident_used++;
                }
            }
            if (slot.prefill_hits > 0 && slot.decode_hits == 0 && slot.prefetch_hits == 0) {
                prefill_only++;
            }
            if (decode_hot_protected(slot)) {
                decode_hot_resident_protected++;
            }
            arc_recent += slot.arc_segment == dstorage_arc_segment::recent ? 1 : 0;
            arc_frequent += slot.arc_segment == dstorage_arc_segment::frequent ? 1 : 0;
        }

        const uint64_t decode_accesses = decode_stats_.hits + decode_stats_.misses;
        const uint64_t prefetch_accesses = prefetch_stats_.hits + prefetch_stats_.misses;
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_SUMMARY slots=%d occupied=%" PRIu64 " decode_touched=%" PRIu64
                " prefill_only=%" PRIu64 " prefetch_touched=%" PRIu64 " prefetch_only=%" PRIu64
                " speculative_slots=%d cache_policy=%s arc_recent=%" PRIu64 " arc_frequent=%" PRIu64
                " pinned_entries=%zu static_pinned_entries=%zu pinned_used_mib=%.2f pinned_budget_mib=%.2f"
                " transfer_ewma_us=%.1f layer_compute_ewma_us=%.1f\n",
                n_slots_, occupied, decode_touched, prefill_only, prefetch_touched, prefetch_only,
                speculative_slots,
                hybrid_arc_policy_ ? "hybrid_arc" : "legacy",
                arc_recent, arc_frequent,
                pinned_cache_.size(), static_pinned_experts_.size(),
                pinned_used_bytes_ / 1024.0 / 1024.0,
                pinned_budget_bytes_ / 1024.0 / 1024.0,
                transfer_ewma_us_, layer_compute_ewma_us_);
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_PHASE prefill_calls=%" PRIu64 " prefill_hits=%" PRIu64 " prefill_misses=%" PRIu64
                " prefill_all_hit_calls=%" PRIu64 " prefill_miss_calls=%" PRIu64
                " decode_calls=%" PRIu64 " decode_hits=%" PRIu64 " decode_misses=%" PRIu64
                " decode_all_hit_calls=%" PRIu64 " decode_miss_calls=%" PRIu64
                " decode_hit_rate=%.4f prefetch_calls=%" PRIu64 " prefetch_hits=%" PRIu64
                " prefetch_misses=%" PRIu64 " prefetch_all_hit_calls=%" PRIu64
                " prefetch_miss_calls=%" PRIu64 " prefetch_hit_rate=%.4f\n",
                prefill_stats_.calls, prefill_stats_.hits, prefill_stats_.misses,
                prefill_stats_.all_hit_calls, prefill_stats_.miss_calls,
                decode_stats_.calls, decode_stats_.hits, decode_stats_.misses,
                decode_stats_.all_hit_calls, decode_stats_.miss_calls,
                decode_accesses > 0 ? double(decode_stats_.hits) / double(decode_accesses) : 0.0,
                prefetch_stats_.calls, prefetch_stats_.hits, prefetch_stats_.misses,
                prefetch_stats_.all_hit_calls, prefetch_stats_.miss_calls,
                prefetch_accesses > 0 ? double(prefetch_stats_.hits) / double(prefetch_accesses) : 0.0);
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_STREAM prefill_calls=%" PRIu64 " prefill_cold_experts=%" PRIu64
                " prefill_pinned_experts=%" PRIu64
                " prefill_requests=%" PRIu64 " prefill_chunks=%" PRIu64
                " prefill_file_mib=%.2f prefill_cuda_mib=%.2f prefill_pinned_mib=%.2f prefill_pinned_us=%" PRIu64
                " prefill_staging_mib=%.2f"
                " prefill_submit_us=%" PRIu64 " prefill_wait_us=%" PRIu64 " prefill_total_us=%" PRIu64
                " decode_calls=%" PRIu64 " decode_cold_experts=%" PRIu64
                " decode_pinned_experts=%" PRIu64
                " decode_possible_runs=%" PRIu64 " decode_possible_run_experts=%" PRIu64
                " decode_runs=%" PRIu64 " decode_run_experts=%" PRIu64
                " decode_requests=%" PRIu64 " decode_chunks=%" PRIu64
                " decode_file_mib=%.2f decode_cuda_mib=%.2f decode_pinned_mib=%.2f decode_pinned_us=%" PRIu64
                " decode_staging_mib=%.2f"
                " decode_submit_us=%" PRIu64 " decode_wait_us=%" PRIu64 " decode_total_us=%" PRIu64
                " decode_max_requests_per_call=%" PRIu64 " decode_max_chunks_per_call=%" PRIu64
                " prefetch_calls=%" PRIu64 " prefetch_cold_experts=%" PRIu64
                " prefetch_pinned_experts=%" PRIu64
                " prefetch_requests=%" PRIu64 " prefetch_chunks=%" PRIu64
                " prefetch_file_mib=%.2f prefetch_cuda_mib=%.2f prefetch_pinned_mib=%.2f prefetch_pinned_us=%" PRIu64
                " prefetch_staging_mib=%.2f"
                " prefetch_submit_us=%" PRIu64 " prefetch_wait_us=%" PRIu64 " prefetch_total_us=%" PRIu64 "\n",
                prefill_stats_.stream_calls,
                prefill_stats_.stream_cold_experts,
                prefill_stats_.stream_pinned_experts,
                prefill_stats_.stream_requests,
                prefill_stats_.stream_chunks,
                prefill_stats_.stream_file_bytes / 1024.0 / 1024.0,
                prefill_stats_.stream_cuda_bytes / 1024.0 / 1024.0,
                prefill_stats_.stream_pinned_bytes / 1024.0 / 1024.0,
                prefill_stats_.stream_pinned_us,
                prefill_stats_.stream_staging_bytes / 1024.0 / 1024.0,
                prefill_stats_.stream_submit_us,
                prefill_stats_.stream_wait_us,
                prefill_stats_.stream_total_us,
                decode_stats_.stream_calls,
                decode_stats_.stream_cold_experts,
                decode_stats_.stream_pinned_experts,
                decode_stats_.stream_possible_runs,
                decode_stats_.stream_possible_run_experts,
                decode_stats_.stream_runs,
                decode_stats_.stream_run_experts,
                decode_stats_.stream_requests,
                decode_stats_.stream_chunks,
                decode_stats_.stream_file_bytes / 1024.0 / 1024.0,
                decode_stats_.stream_cuda_bytes / 1024.0 / 1024.0,
                decode_stats_.stream_pinned_bytes / 1024.0 / 1024.0,
                decode_stats_.stream_pinned_us,
                decode_stats_.stream_staging_bytes / 1024.0 / 1024.0,
                decode_stats_.stream_submit_us,
                decode_stats_.stream_wait_us,
                decode_stats_.stream_total_us,
                decode_stats_.stream_max_requests_per_call,
                decode_stats_.stream_max_chunks_per_call,
                prefetch_stats_.stream_calls,
                prefetch_stats_.stream_cold_experts,
                prefetch_stats_.stream_pinned_experts,
                prefetch_stats_.stream_requests,
                prefetch_stats_.stream_chunks,
                prefetch_stats_.stream_file_bytes / 1024.0 / 1024.0,
                prefetch_stats_.stream_cuda_bytes / 1024.0 / 1024.0,
                prefetch_stats_.stream_pinned_bytes / 1024.0 / 1024.0,
                prefetch_stats_.stream_pinned_us,
                prefetch_stats_.stream_staging_bytes / 1024.0 / 1024.0,
                prefetch_stats_.stream_submit_us,
                prefetch_stats_.stream_wait_us,
                prefetch_stats_.stream_total_us);
        auto print_io_shape = [](const char * phase_name, const PhaseStats & stats) {
            const uint64_t requests = stats.stream_requests;
            const uint64_t pairs = requests > stats.stream_calls ? requests - stats.stream_calls : 0;
            const uint64_t min_request =
                    stats.stream_min_request_bytes == UINT64_MAX ? 0 : stats.stream_min_request_bytes;
            const double file_mib = stats.stream_file_bytes / 1024.0 / 1024.0;
            const double total_s = stats.stream_total_us / 1000000.0;
            const double effective_mib_s = total_s > 0.0 ? file_mib / total_s : 0.0;
            const double avg_request_kib =
                    requests > 0 ? double(stats.stream_file_bytes) / double(requests) / 1024.0 : 0.0;
            const double aligned_ratio =
                    requests > 0 ? double(stats.stream_aligned_4k_requests) / double(requests) : 0.0;
            const double contiguous_ratio =
                    pairs > 0 ? double(stats.stream_contiguous_request_pairs) / double(pairs) : 0.0;
            const uint64_t file_offset_mod_min =
                    stats.stream_file_offset_mod_min == UINT64_MAX ? 0 : stats.stream_file_offset_mod_min;
            const uint64_t size_mod_min =
                    stats.stream_size_mod_min == UINT64_MAX ? 0 : stats.stream_size_mod_min;
            const uint64_t cuda_dest_mod_min =
                    stats.stream_cuda_dest_mod_min == UINT64_MAX ? 0 : stats.stream_cuda_dest_mod_min;
            const uint64_t uncompressed_mod_min =
                    stats.stream_uncompressed_mod_min == UINT64_MAX ? 0 : stats.stream_uncompressed_mod_min;
            std::fprintf(
                    stderr,
                    "DSTORAGE_MOE_IO_SHAPE phase=%s requests=%" PRIu64 " chunks=%" PRIu64
                    " avg_request_kib=%.2f min_request_kib=%.2f max_request_kib=%.2f"
                    " aligned_4k=%" PRIu64 " unaligned_4k=%" PRIu64 " aligned_ratio=%.4f"
                    " offset_aligned_4k=%" PRIu64 " size_aligned_4k=%" PRIu64
                    " dest_aligned_4k=%" PRIu64 " uncompressed_aligned_4k=%" PRIu64
                    " offset_mod_min=%" PRIu64 " offset_mod_max=%" PRIu64 " offset_mod_gcd=%" PRIu64
                    " size_mod_min=%" PRIu64 " size_mod_max=%" PRIu64 " size_mod_gcd=%" PRIu64
                    " dest_mod_min=%" PRIu64 " dest_mod_max=%" PRIu64 " dest_mod_gcd=%" PRIu64
                    " uncompressed_mod_min=%" PRIu64 " uncompressed_mod_max=%" PRIu64
                    " uncompressed_mod_gcd=%" PRIu64
                    " contiguous_pairs=%" PRIu64 " contiguous_ratio=%.4f"
                    " backward_or_overlap_pairs=%" PRIu64 " forward_gap_mib=%.2f"
                    " size_le_64k=%" PRIu64 " size_le_256k=%" PRIu64
                    " size_le_1m=%" PRIu64 " size_le_4m=%" PRIu64 " size_gt_4m=%" PRIu64
                    " max_chunk_staging_mib=%.2f effective_mib_s=%.2f\n",
                    phase_name,
                    requests,
                    stats.stream_chunks,
                    avg_request_kib,
                    min_request / 1024.0,
                    stats.stream_max_request_bytes / 1024.0,
                    stats.stream_aligned_4k_requests,
                    stats.stream_unaligned_4k_requests,
                    aligned_ratio,
                    stats.stream_file_offset_4k_aligned,
                    stats.stream_size_4k_aligned,
                    stats.stream_cuda_dest_4k_aligned,
                    stats.stream_uncompressed_4k_aligned,
                    file_offset_mod_min,
                    stats.stream_file_offset_mod_max,
                    stats.stream_file_offset_mod_gcd,
                    size_mod_min,
                    stats.stream_size_mod_max,
                    stats.stream_size_mod_gcd,
                    cuda_dest_mod_min,
                    stats.stream_cuda_dest_mod_max,
                    stats.stream_cuda_dest_mod_gcd,
                    uncompressed_mod_min,
                    stats.stream_uncompressed_mod_max,
                    stats.stream_uncompressed_mod_gcd,
                    stats.stream_contiguous_request_pairs,
                    contiguous_ratio,
                    stats.stream_backward_or_overlap_pairs,
                    stats.stream_forward_gap_bytes / 1024.0 / 1024.0,
                    stats.stream_size_le_64k,
                    stats.stream_size_le_256k,
                    stats.stream_size_le_1m,
                    stats.stream_size_le_4m,
                    stats.stream_size_gt_4m,
                    stats.stream_max_chunk_staging_bytes / 1024.0 / 1024.0,
                    effective_mib_s);
        };
        print_io_shape("prefill", prefill_stats_);
        print_io_shape("decode", decode_stats_);
        print_io_shape("prefetch", prefetch_stats_);
        auto print_pool_align = [&](const char * name, const std::unordered_map<std::string, TensorTypePool> & pools) {
            uint64_t base_4k = 0;
            uint64_t stride_4k = 0;
            uint64_t both_4k = 0;
            uint64_t base_mod_gcd = 0;
            uint64_t stride_mod_gcd = 0;
            uint64_t base_mod_min = UINT64_MAX;
            uint64_t base_mod_max = 0;
            uint64_t stride_mod_min = UINT64_MAX;
            uint64_t stride_mod_max = 0;
            for (const auto & [key, pool] : pools) {
                GGML_UNUSED(key);
                const uint64_t base_mod = pool.cuda_ptr % 4096;
                const uint64_t stride_mod = pool.slot_size % 4096;
                base_4k += base_mod == 0 ? 1 : 0;
                stride_4k += stride_mod == 0 ? 1 : 0;
                both_4k += (base_mod == 0 && stride_mod == 0) ? 1 : 0;
                if (base_mod != 0) {
                    base_mod_min = std::min(base_mod_min, base_mod);
                    base_mod_max = std::max(base_mod_max, base_mod);
                    base_mod_gcd = base_mod_gcd == 0 ? base_mod : std::gcd(base_mod_gcd, base_mod);
                }
                if (stride_mod != 0) {
                    stride_mod_min = std::min(stride_mod_min, stride_mod);
                    stride_mod_max = std::max(stride_mod_max, stride_mod);
                    stride_mod_gcd = stride_mod_gcd == 0 ? stride_mod : std::gcd(stride_mod_gcd, stride_mod);
                }
            }
            std::fprintf(
                    stderr,
                    "DSTORAGE_MOE_POOL_ALIGN kind=%s align_pool_4k=%d align_stride_4k_unsafe=%d pools=%zu"
                    " base_4k=%" PRIu64 " stride_4k=%" PRIu64 " both_4k=%" PRIu64
                    " base_mod_min=%" PRIu64 " base_mod_max=%" PRIu64 " base_mod_gcd=%" PRIu64
                    " stride_mod_min=%" PRIu64 " stride_mod_max=%" PRIu64 " stride_mod_gcd=%" PRIu64 "\n",
                    name,
                    llama_dstorage_align_pool_base_4k() ? 1 : 0,
                    llama_dstorage_align_pool_stride_4k() ? 1 : 0,
                    pools.size(),
                    base_4k,
                    stride_4k,
                    both_4k,
                    base_mod_min == UINT64_MAX ? 0 : base_mod_min,
                    base_mod_max,
                    base_mod_gcd,
                    stride_mod_min == UINT64_MAX ? 0 : stride_mod_min,
                    stride_mod_max,
                    stride_mod_gcd);
        };
        print_pool_align("persistent", type_pools_);
        print_pool_align("active", active_type_pools_);
#if LLAMA_DSTORAGE_HAS_BACKEND
        if (ds_loader_ != nullptr) {
            DSLoaderIoStats io_stats = {};
            if (ds_loader_get_io_stats(ds_loader_, &io_stats) == 0) {
                const double mib = io_stats.bytes / 1024.0 / 1024.0;
                const double wall_s = io_stats.wall_us / 1000000.0;
                const double read_s = io_stats.read_sum_us / 1000000.0;
                const double avg_request_kib =
                        io_stats.requests > 0 ? double(io_stats.bytes) / double(io_stats.requests) / 1024.0 : 0.0;
                const double wall_mib_s = wall_s > 0.0 ? mib / wall_s : 0.0;
                const double read_mib_s = read_s > 0.0 ? mib / read_s : 0.0;
                const double effective_concurrency =
                        io_stats.wall_us > 0 ? double(io_stats.read_sum_us) / double(io_stats.wall_us) : 0.0;
                std::fprintf(
                        stderr,
                        "DSTORAGE_GDS_BACKEND batches=%" PRIu64 " requests=%" PRIu64 " bytes_mib=%.2f"
                        " avg_request_kib=%.2f min_request_kib=%.2f max_request_kib=%.2f"
                        " wall_us=%" PRIu64 " read_sum_us=%" PRIu64 " read_max_us=%" PRIu64
                        " wall_mib_s=%.2f read_mib_s=%.2f effective_concurrency=%.2f"
                        " batch_api_attempts=%" PRIu64 " batch_api_successes=%" PRIu64
                        " worker_batches=%" PRIu64 " serial_batches=%" PRIu64
                        " aligned_4k=%" PRIu64 " unaligned_4k=%" PRIu64
                        " offset_aligned_4k=%" PRIu64 " size_aligned_4k=%" PRIu64
                        " dest_aligned_4k=%" PRIu64
                        " offset_mod_min=%" PRIu64 " offset_mod_max=%" PRIu64 " offset_mod_gcd=%" PRIu64
                        " size_mod_min=%" PRIu64 " size_mod_max=%" PRIu64 " size_mod_gcd=%" PRIu64
                        " dest_mod_min=%" PRIu64 " dest_mod_max=%" PRIu64 " dest_mod_gcd=%" PRIu64
                        " size_le_64k=%" PRIu64 " size_le_256k=%" PRIu64
                        " size_le_1m=%" PRIu64 " size_le_4m=%" PRIu64 " size_gt_4m=%" PRIu64
                        " max_batch_requests=%" PRIu64 " max_batch_mib=%.2f max_batch_wall_us=%" PRIu64 "\n",
                        io_stats.batches,
                        io_stats.requests,
                        mib,
                        avg_request_kib,
                        io_stats.min_request_bytes / 1024.0,
                        io_stats.max_request_bytes / 1024.0,
                        io_stats.wall_us,
                        io_stats.read_sum_us,
                        io_stats.read_max_us,
                        wall_mib_s,
                        read_mib_s,
                        effective_concurrency,
                        io_stats.batch_api_attempts,
                        io_stats.batch_api_successes,
                        io_stats.worker_batches,
                        io_stats.serial_batches,
                        io_stats.aligned_4k_requests,
                        io_stats.unaligned_4k_requests,
                        io_stats.file_offset_4k_aligned,
                        io_stats.size_4k_aligned,
                        io_stats.cuda_dest_4k_aligned,
                        io_stats.file_offset_mod_min,
                        io_stats.file_offset_mod_max,
                        io_stats.file_offset_mod_gcd,
                        io_stats.size_mod_min,
                        io_stats.size_mod_max,
                        io_stats.size_mod_gcd,
                        io_stats.cuda_dest_mod_min,
                        io_stats.cuda_dest_mod_max,
                        io_stats.cuda_dest_mod_gcd,
                        io_stats.size_le_64k,
                        io_stats.size_le_256k,
                        io_stats.size_le_1m,
                        io_stats.size_le_4m,
                        io_stats.size_gt_4m,
                        io_stats.max_batch_requests,
                        io_stats.max_batch_bytes / 1024.0 / 1024.0,
                        io_stats.max_batch_wall_us);
            }
        }
#endif
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_PREFETCH candidate_calls=%" PRIu64 " reject_observations=%" PRIu64
                " reject_resident=%" PRIu64 " reject_policy=%" PRIu64
                " submitted_calls=%" PRIu64 " submitted_experts=%" PRIu64
                " completed_ok=%" PRIu64 " completed_failed=%" PRIu64
                " loaded_experts=%" PRIu64 " useful_hits=%" PRIu64
                " useful_prefill_hits=%" PRIu64 " useful_decode_hits=%" PRIu64
                " evicted_before_use=%" PRIu64 " evicted_after_use=%" PRIu64
                " resident_unused=%" PRIu64 " resident_used=%" PRIu64
                " wait_calls=%" PRIu64 " wait_us=%" PRIu64
                " best_confidence=%.4f best_candidate_value=%" PRId64
                " lowest_victim_cost=%" PRId64 "\n",
                diagnostic_stats_.prefetch_candidate_calls,
                diagnostic_stats_.prefetch_reject_observations,
                diagnostic_stats_.prefetch_reject_resident,
                diagnostic_stats_.prefetch_reject_policy,
                diagnostic_stats_.prefetch_submitted_calls,
                diagnostic_stats_.prefetch_submitted_experts,
                diagnostic_stats_.prefetch_completed_ok,
                diagnostic_stats_.prefetch_completed_failed,
                diagnostic_stats_.prefetch_loaded_experts,
                diagnostic_stats_.useful_prefetch_hits,
                diagnostic_stats_.useful_prefetch_prefill_hits,
                diagnostic_stats_.useful_prefetch_decode_hits,
                diagnostic_stats_.prefetch_evicted_before_use,
                diagnostic_stats_.prefetch_evicted_after_use,
                prefetch_resident_unused,
                prefetch_resident_used,
                diagnostic_stats_.prefetch_wait_calls,
                diagnostic_stats_.prefetch_wait_us,
                diagnostic_stats_.prefetch_best_confidence,
                diagnostic_stats_.prefetch_best_candidate_value,
                diagnostic_stats_.prefetch_lowest_victim_cost == std::numeric_limits<int64_t>::max()
                        ? 0
                        : diagnostic_stats_.prefetch_lowest_victim_cost);
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_PINNED budget_mib=%.2f used_mib=%.2f entries=%zu"
                " queued=%" PRIu64 " completed=%" PRIu64 " dropped=%" PRIu64
                " admit_mib=%.2f admit_us=%" PRIu64
                " cuda_pinned=%" PRIu64 " fallback_ram=%" PRIu64
                " pending_mib=%.2f skipped_policy=%" PRIu64
                " skipped_pending=%" PRIu64 " skipped_budget=%" PRIu64
                " skipped_throttle=%" PRIu64
                " prefill_only_releases=%" PRIu64
                " prefill_only_release_entries=%" PRIu64
                " prefill_only_release_mib=%.2f"
                " prefill_only_release_us=%" PRIu64 "\n",
                pinned_budget_bytes_ / 1024.0 / 1024.0,
                pinned_used_bytes_ / 1024.0 / 1024.0,
                pinned_cache_.size(),
                diagnostic_stats_.pinned_admit_queued,
                diagnostic_stats_.pinned_admit_completed,
                diagnostic_stats_.pinned_admit_dropped,
                diagnostic_stats_.pinned_admit_bytes / 1024.0 / 1024.0,
                diagnostic_stats_.pinned_admit_us,
                diagnostic_stats_.pinned_admit_cuda_pinned,
                diagnostic_stats_.pinned_admit_fallback_ram,
                pinned_pending_bytes_ / 1024.0 / 1024.0,
                diagnostic_stats_.pinned_admit_skipped_policy,
                diagnostic_stats_.pinned_admit_skipped_pending,
                diagnostic_stats_.pinned_admit_skipped_budget,
                diagnostic_stats_.pinned_admit_skipped_throttle,
                diagnostic_stats_.pinned_prefill_only_releases,
                diagnostic_stats_.pinned_prefill_only_release_entries,
                diagnostic_stats_.pinned_prefill_only_release_bytes / 1024.0 / 1024.0,
                diagnostic_stats_.pinned_prefill_only_release_us);
        const uint64_t decode_hot_total = decode_hot_stats_.hits + decode_hot_stats_.misses;
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_DECODE_HOT enabled=%d slot_cap=%d min_hits=%d"
                " rebuild_interval=%d"
                " protected_entries=%zu resident_protected=%" PRIu64
                " accesses=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64
                " hit_rate=%.4f protected_hits=%" PRIu64
                " protected_misses=%" PRIu64 " protected_evictions=%" PRIu64
                " rebuilds=%" PRIu64 "\n",
                llama_dstorage_decode_hot_cache_enabled() ? 1 : 0,
                llama_dstorage_decode_hot_cache_slots(),
                llama_dstorage_decode_hot_min_hits(),
                llama_dstorage_decode_hot_rebuild_interval(),
                decode_hot_protected_.size(),
                decode_hot_resident_protected,
                decode_hot_stats_.accesses,
                decode_hot_stats_.hits,
                decode_hot_stats_.misses,
                decode_hot_total > 0
                        ? double(decode_hot_stats_.hits) / double(decode_hot_total)
                        : 0.0,
                decode_hot_stats_.protected_hits,
                decode_hot_stats_.protected_misses,
                decode_hot_stats_.protected_evictions,
                decode_hot_stats_.rebuilds);
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_EVICT prefill_evicted_prefill_only=%" PRIu64
                " prefill_evicted_decode_touched=%" PRIu64
                " decode_evicted_prefill_only=%" PRIu64
                " decode_evicted_decode_touched=%" PRIu64
                " prefetch_evicted_prefill_only=%" PRIu64
                " prefetch_evicted_decode_touched=%" PRIu64 "\n",
                prefill_stats_.evicted_prefill_only,
                prefill_stats_.evicted_decode_touched,
                decode_stats_.evicted_prefill_only,
                decode_stats_.evicted_decode_touched,
                prefetch_stats_.evicted_prefill_only,
                prefetch_stats_.evicted_decode_touched);
        const double timeline_hit_rate =
                timeline_stats_.unique_experts > 0
                        ? double(timeline_stats_.hit_experts) / double(timeline_stats_.unique_experts)
                        : 0.0;
        const double avg_ensure_us =
                timeline_stats_.real_calls > 0
                        ? double(timeline_stats_.ensure_total_us) / double(timeline_stats_.real_calls)
                        : 0.0;
        const double avg_stream_us =
                timeline_stats_.miss_calls > 0
                        ? double(timeline_stats_.stream_total_us) / double(timeline_stats_.miss_calls)
                        : 0.0;
        const double avg_compute_us =
                timeline_stats_.compute_samples > 0
                        ? double(timeline_stats_.compute_total_us) / double(timeline_stats_.compute_samples)
                        : 0.0;
        const uint64_t compute_min_us =
                timeline_stats_.compute_min_us == std::numeric_limits<uint64_t>::max()
                        ? 0
                        : timeline_stats_.compute_min_us;
        std::fprintf(
                stderr,
                "DSTORAGE_MOE_TIMELINE real_calls=%" PRIu64
                " prefill_calls=%" PRIu64 " decode_calls=%" PRIu64
                " all_hit_calls=%" PRIu64 " miss_calls=%" PRIu64
                " selected_experts=%" PRIu64 " unique_experts=%" PRIu64
                " hit_experts=%" PRIu64 " miss_experts=%" PRIu64
                " hit_rate=%.4f ensure_total_us=%" PRIu64
                " avg_ensure_us=%.1f stream_total_us=%" PRIu64
                " avg_stream_miss_us=%.1f stream_file_mib=%.2f"
                " compute_samples=%" PRIu64 " compute_avg_us=%.1f"
                " compute_min_us=%" PRIu64 " compute_max_us=%" PRIu64 "\n",
                timeline_stats_.real_calls,
                timeline_stats_.prefill_calls,
                timeline_stats_.decode_calls,
                timeline_stats_.all_hit_calls,
                timeline_stats_.miss_calls,
                timeline_stats_.selected_experts,
                timeline_stats_.unique_experts,
                timeline_stats_.hit_experts,
                timeline_stats_.miss_experts,
                timeline_hit_rate,
                timeline_stats_.ensure_total_us,
                avg_ensure_us,
                timeline_stats_.stream_total_us,
                avg_stream_us,
                timeline_stats_.stream_file_bytes / 1024.0 / 1024.0,
                timeline_stats_.compute_samples,
                avg_compute_us,
                compute_min_us,
                timeline_stats_.compute_max_us);
        {
            std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> steps;
            {
                std::lock_guard<std::mutex> lock(g_ensure_step_profile.mutex);
                steps.assign(g_ensure_step_profile.steps.begin(), g_ensure_step_profile.steps.end());
            }
            std::sort(steps.begin(), steps.end(),
                    [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
            for (const auto & s : steps) {
                std::fprintf(stderr,
                        "DSTORAGE_ENSURE_PROFILE step=%s total_us=%" PRIu64 " calls=%" PRIu64
                        " avg_us=%.1f\n",
                        s.first.c_str(), s.second.first, s.second.second,
                        s.second.second ? double(s.second.first) / double(s.second.second) : 0.0);
            }
        }
        auto print_dry_run = [](const char * label, const SchedulerDryRunStats & stats) {
            const double dry_hidden_ratio =
                    stats.transfer_total_us > 0
                            ? double(stats.hidden_possible_us) / double(stats.transfer_total_us)
                            : 0.0;
            const double dry_file_fit_ratio =
                    stats.file_bytes > 0
                            ? double(stats.hideable_file_bytes) / double(stats.file_bytes)
                            : 0.0;
            const int64_t best_margin_us =
                    stats.best_margin_us == std::numeric_limits<int64_t>::min()
                            ? 0
                            : stats.best_margin_us;
            const int64_t worst_margin_us =
                    stats.worst_margin_us == std::numeric_limits<int64_t>::max()
                            ? 0
                            : stats.worst_margin_us;
            std::fprintf(
                    stderr,
                    "DSTORAGE_MOE_SCHED_DRYRUN_%s miss_calls=%" PRIu64
                    " no_window_calls=%" PRIu64 " full_hide_calls=%" PRIu64
                    " partial_hide_calls=%" PRIu64
                    " transfer_total_us=%" PRIu64 " prior_compute_total_us=%" PRIu64
                    " hidden_possible_us=%" PRIu64 " visible_after_overlap_us=%" PRIu64
                    " hidden_ratio=%.4f file_mib=%.2f hideable_file_mib=%.2f"
                    " file_fit_ratio=%.4f best_margin_us=%" PRId64
                    " worst_margin_us=%" PRId64 "\n",
                    label,
                    stats.miss_calls,
                    stats.no_window_calls,
                    stats.full_hide_calls,
                    stats.partial_hide_calls,
                    stats.transfer_total_us,
                    stats.prior_compute_total_us,
                    stats.hidden_possible_us,
                    stats.visible_after_overlap_us,
                    dry_hidden_ratio,
                    stats.file_bytes / 1024.0 / 1024.0,
                    stats.hideable_file_bytes / 1024.0 / 1024.0,
                    dry_file_fit_ratio,
                    best_margin_us,
                    worst_margin_us);
        };
        print_dry_run("ALL", scheduler_dry_run_stats_);
        print_dry_run("PREFILL", scheduler_prefill_dry_run_stats_);
        print_dry_run("DECODE", scheduler_decode_dry_run_stats_);
        std::fflush(stderr);
    }
    destroy_pinned_cache();
#if LLAMA_DSTORAGE_HAS_BACKEND
    for (auto & [key, pool] : type_pools_) {
        if (pool.cuda_ptr != 0 && pool.owns_alloc) {
            ds_loader_cuda_free(pool.alloc_ptr != 0 ? pool.alloc_ptr : pool.cuda_ptr);
            pool.cuda_ptr = 0;
            pool.alloc_ptr = 0;
        }
    }
    for (auto & [key, pool] : active_type_pools_) {
        if (pool.cuda_ptr != 0 && pool.owns_alloc) {
            ds_loader_cuda_free(pool.alloc_ptr != 0 ? pool.alloc_ptr : pool.cuda_ptr);
            pool.cuda_ptr = 0;
            pool.alloc_ptr = 0;
        }
    }
    if (bundle_staging_ptr_ != 0) {
        ds_loader_cuda_free(bundle_staging_ptr_);
        bundle_staging_ptr_ = 0;
        bundle_staging_size_ = 0;
    }
    if (ds_loader_ != nullptr) {
        ds_loader_destroy(ds_loader_);
        ds_loader_ = nullptr;
    }
#endif
    type_pools_.clear();
    active_type_pools_.clear();
    active_capacity_ = 0;
    active_slot_count_ = 0;
    execution_slots_by_layer_.clear();
    slots_.clear();
    pool_group_slot_counts_.clear();
    layer_pool_groups_.clear();
    layer_arc_.clear();
    resident_experts_.clear();
    tensor_registry_.clear();
    layer_tensors_.clear();
    type_strides_.clear();
    layer_reload_bytes_.clear();
    n_slots_ = 0;
    pools_allocated_ = false;
    use_tick_ = 0;
}

DStorageSlotManager::~DStorageSlotManager() {
    destroy();
}

const DStorageSlotManager::PinnedEntry * DStorageSlotManager::get_pinned_entry(int layer_idx, int expert_idx) {
    std::lock_guard<std::mutex> slot_lock(slots_mutex_);
    uint64_t expert_key = make_expert_key(layer_idx, expert_idx);
    auto it = pinned_cache_.find(expert_key);
    if (it != pinned_cache_.end()) {
        return &it->second;
    }
    return nullptr;
}
