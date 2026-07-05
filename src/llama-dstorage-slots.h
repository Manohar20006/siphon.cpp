#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static inline bool llama_dstorage_debug_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_DSTORAGE_DEBUG");
        return value != nullptr &&
               std::strcmp(value, "") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0;
    }();
    return enabled;
}

#define LLAMA_DSTORAGE_DEBUG_LOG(...) \
    do { \
        if (llama_dstorage_debug_enabled()) { \
            std::fprintf(stderr, __VA_ARGS__); \
            std::fflush(stderr); \
        } \
    } while (0)

struct DSLoader;
typedef struct DSLoader * DSLoaderHandle;

enum class dstorage_moe_phase {
    prefill,
    decode,
    prefetch,
};

enum class dstorage_arc_segment : uint8_t {
    none,
    recent,
    frequent,
};

static inline const char * llama_dstorage_phase_name(dstorage_moe_phase phase) {
    switch (phase) {
        case dstorage_moe_phase::decode:   return "decode";
        case dstorage_moe_phase::prefetch: return "prefetch";
        case dstorage_moe_phase::prefill:
        default:                           return "prefill";
    }
}

// Information about a registered expert tensor (gate_up_exps or down_exps)
struct ExpertTensorInfo {
    std::wstring file_path;
    uint64_t     file_offset     = 0;   // byte offset in the GGUF file for the WHOLE tensor
    uint64_t     total_size      = 0;   // total bytes of ALL experts combined
    uint64_t     expert_stride   = 0;   // bytes per single expert = total_size / n_experts
    int          n_experts       = 0;
    int          type_size_row   = 0;   // ggml type element size for this tensor
    bool         sidecar_expert_major = false;
    uint64_t     sidecar_record_stride = 0;
    uint64_t     sidecar_tensor_offset = 0;
    
    // Decompression metadata
    uint64_t     total_size_compressed = 0;
    std::vector<uint32_t> expert_sizes_comp; // size of each compressed expert
};

// A single slot in the shared slot table — tracks which expert occupies it.
// The actual CUDA addresses are computed on the fly from per-type pools.
struct ExpertSlot {
    int  layer_idx   = -1;
    int  expert_idx  = -1;
    int  pool_group  = 0;
    int  pool_index  = -1;
    bool occupied    = false;
    bool admission_pending = false;
    bool transfer_pending = false;
    uint32_t execution_pins = 0;
    uint64_t last_used_tick = 0;
    uint32_t prefill_hits = 0;
    uint32_t decode_hits  = 0;
    uint32_t prefetch_hits = 0;
    dstorage_arc_segment arc_segment = dstorage_arc_segment::none;
    uint8_t contiguous_neighbors = 0;
};

class DStorageSlotManager {
public:
    struct CacheLayout {
        uint64_t persistent_cache_budget_bytes = 0;
        uint64_t workspace_reserve_bytes = 0;
        int n_persistent_slots = 0;
    };

    // Initialize the slot manager.
    // moe_gpu_cache_mib: global VRAM budget for cached expert slots.
    // moe_pinned_cache_mib: optional pinned host RAM budget for L2 expert cache.
    bool init(
        int n_layers,
        int n_experts,
        int n_expert_used,
        uint32_t moe_gpu_cache_mib,
        uint32_t moe_pinned_cache_mib = 0,
        bool phase_cache_policy = false);

    static int calculate_slot_count(
        uint64_t pool_budget_bytes,
        uint64_t bytes_per_logical_slot,
        int n_layers,
        int n_experts) {
        if (pool_budget_bytes == 0 || bytes_per_logical_slot == 0 || n_layers <= 0 || n_experts <= 0) {
            return 0;
        }

        const uint64_t raw_slots = pool_budget_bytes / bytes_per_logical_slot;
        const uint64_t useful_slots = uint64_t(n_layers) * uint64_t(n_experts);
        const uint64_t capped_slots = std::min(raw_slots, useful_slots);
        return capped_slots > uint64_t(std::numeric_limits<int>::max()) ? std::numeric_limits<int>::max() : int(capped_slots);
    }

    static CacheLayout calculate_cache_layout(
        uint64_t persistent_cache_budget_bytes,
        uint64_t bytes_per_logical_slot,
        int n_layers,
        int n_experts) {
        CacheLayout layout;
        layout.persistent_cache_budget_bytes = persistent_cache_budget_bytes;
        layout.workspace_reserve_bytes = 0;
        layout.n_persistent_slots = calculate_slot_count(
                layout.persistent_cache_budget_bytes,
                bytes_per_logical_slot,
                n_layers,
                n_experts);
        return layout;
    }

    static uint32_t calculate_phase_cache_mib(
            uint32_t requested_cache_mib,
            bool phase_cache_policy,
            bool prefill) {
        if (!phase_cache_policy || !prefill) {
            return requested_cache_mib;
        }

        uint32_t prefill_mib = std::min(1000u, requested_cache_mib);
        if (prefill_mib < 512) {
            prefill_mib = std::min(requested_cache_mib, 512u);
        }
        return prefill_mib;
    }

    static bool should_reallocate_for_phase(
            bool phase_cache_policy,
            dstorage_moe_phase current_phase,
            dstorage_moe_phase target_phase,
            uint64_t prefill_budget,
            uint64_t decode_budget) {
        if (!phase_cache_policy || current_phase == target_phase) {
            return false;
        }

        const uint64_t current_budget =
                current_phase == dstorage_moe_phase::prefill ? prefill_budget : decode_budget;
        const uint64_t target_budget =
                target_phase == dstorage_moe_phase::prefill ? prefill_budget : decode_budget;
        return current_budget != target_budget;
    }

    static int calculate_decode_workspace_capacity(int n_unique_experts, int n_expert_used) {
        if (n_unique_experts <= 0 || n_expert_used <= 0 || n_unique_experts > n_expert_used) {
            return 0;
        }
        return n_unique_experts;
    }

    static int calculate_prefill_token_chunk(int n_expert_used, int max_workspace_experts) {
        if (n_expert_used <= 0 || max_workspace_experts <= 0) {
            return 1;
        }
        return std::max(1, max_workspace_experts / n_expert_used);
    }

    static constexpr int max_prefill_workspace_experts() {
        return 192;
    }

    class RequestActivationStats {
    public:
        void begin_request() {
            prefill_counts_.clear();
            decode_counts_.clear();
            decode_observations_.clear();
            generation_++;
        }

        void record(
                int layer_idx,
                const int32_t * expert_ids,
                int n_selected,
                dstorage_moe_phase phase,
                bool decode_like_for_history = false) {
            if (expert_ids == nullptr || n_selected <= 0 || phase == dstorage_moe_phase::prefetch) {
                return;
            }

            const bool decode_like = phase == dstorage_moe_phase::decode || decode_like_for_history;
            auto & counts = decode_like ? decode_counts_ : prefill_counts_;
            if (decode_like) {
                decode_observations_[layer_idx]++;
                rolling_decode_observations_[layer_idx]++;
            }
            for (int i = 0; i < n_selected; ++i) {
                if (expert_ids[i] >= 0) {
                    counts[key(layer_idx, expert_ids[i])]++;
                    if (decode_like) {
                        rolling_decode_counts_[key(layer_idx, expert_ids[i])]++;
                    }
                }
            }
        }

        uint32_t count(int layer_idx, int expert_idx, dstorage_moe_phase phase) const {
            const auto & counts = phase == dstorage_moe_phase::decode ? decode_counts_ : prefill_counts_;
            const auto it = counts.find(key(layer_idx, expert_idx));
            return it != counts.end() ? it->second : 0;
        }

        uint64_t generation() const {
            return generation_;
        }

        uint32_t decode_observations(int layer_idx) const {
            const auto it = decode_observations_.find(layer_idx);
            return it != decode_observations_.end() ? it->second : 0;
        }

        double decode_confidence(int layer_idx, int expert_idx) const {
            const uint32_t observations = decode_observations(layer_idx);
            return observations > 0
                    ? double(count(layer_idx, expert_idx, dstorage_moe_phase::decode)) / observations
                    : 0.0;
        }

        uint32_t rolling_decode_count(int layer_idx, int expert_idx) const {
            const auto it = rolling_decode_counts_.find(key(layer_idx, expert_idx));
            return it != rolling_decode_counts_.end() ? it->second : 0;
        }

        uint32_t rolling_decode_observations(int layer_idx) const {
            const auto it = rolling_decode_observations_.find(layer_idx);
            return it != rolling_decode_observations_.end() ? it->second : 0;
        }

        double rolling_decode_confidence(int layer_idx, int expert_idx) const {
            const uint32_t observations = rolling_decode_observations(layer_idx);
            return observations > 0
                    ? double(rolling_decode_count(layer_idx, expert_idx)) / observations
                    : 0.0;
        }

        std::vector<int32_t> top_decode_experts(
                int layer_idx,
                int max_experts,
                double min_confidence) const {
            std::vector<std::pair<int32_t, uint32_t>> ranked;
            if (max_experts <= 0) {
                return {};
            }

            const uint32_t observations = decode_observations(layer_idx);
            if (observations == 0) {
                return {};
            }

            for (const auto & entry : decode_counts_) {
                const int entry_layer = int(uint32_t(entry.first >> 32));
                if (entry_layer != layer_idx) {
                    continue;
                }
                const double confidence = double(entry.second) / double(observations);
                if (confidence >= min_confidence) {
                    ranked.emplace_back(int32_t(uint32_t(entry.first)), entry.second);
                }
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

            std::vector<int32_t> experts;
            experts.reserve(std::min<int>(max_experts, int(ranked.size())));
            for (const auto & entry : ranked) {
                if (int(experts.size()) >= max_experts) {
                    break;
                }
                experts.push_back(entry.first);
            }
            return experts;
        }

        std::vector<int32_t> top_rolling_decode_experts(
                int layer_idx,
                int max_experts,
                double min_confidence,
                uint32_t min_observations) const {
            std::vector<std::pair<int32_t, uint32_t>> ranked;
            if (max_experts <= 0) {
                return {};
            }

            const uint32_t observations = rolling_decode_observations(layer_idx);
            if (observations < min_observations) {
                return {};
            }

            for (const auto & entry : rolling_decode_counts_) {
                const int entry_layer = int(uint32_t(entry.first >> 32));
                if (entry_layer != layer_idx) {
                    continue;
                }
                const double confidence = double(entry.second) / double(observations);
                if (confidence >= min_confidence) {
                    ranked.emplace_back(int32_t(uint32_t(entry.first)), entry.second);
                }
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

            std::vector<int32_t> experts;
            experts.reserve(std::min<int>(max_experts, int(ranked.size())));
            for (const auto & entry : ranked) {
                if (int(experts.size()) >= max_experts) {
                    break;
                }
                experts.push_back(entry.first);
            }
            return experts;
        }

    private:
        static uint64_t key(int layer_idx, int expert_idx) {
            return (uint64_t(uint32_t(layer_idx)) << 32) | uint32_t(expert_idx);
        }

        std::unordered_map<uint64_t, uint32_t> prefill_counts_;
        std::unordered_map<uint64_t, uint32_t> decode_counts_;
        std::unordered_map<int, uint32_t> decode_observations_;
        std::unordered_map<uint64_t, uint32_t> rolling_decode_counts_;
        std::unordered_map<int, uint32_t> rolling_decode_observations_;
        uint64_t generation_ = 0;
    };

    static bool is_better_eviction_candidate(const ExpertSlot & candidate, const ExpertSlot & current_best) {
        // 1. Prefer evicting slots that are NOT decode touched
        const bool candidate_decode_touched = candidate.decode_hits > 0;
        const bool best_decode_touched = current_best.decode_hits > 0;
        if (candidate_decode_touched != best_decode_touched) {
            return !candidate_decode_touched;
        }

        // 2. If both are decode-touched, prefer evicting the one with FEWER hits (lower frequency)
        if (candidate_decode_touched) {
            if (candidate.decode_hits != current_best.decode_hits) {
                return candidate.decode_hits < current_best.decode_hits;
            }
        } else {
            // If neither is decode touched, check prefill hits
            const bool candidate_prefill_touched = candidate.prefill_hits > 0;
            const bool best_prefill_touched = current_best.prefill_hits > 0;
            if (candidate_prefill_touched != best_prefill_touched) {
                return !candidate_prefill_touched;
            }
            if (candidate_prefill_touched) {
                if (candidate.prefill_hits != current_best.prefill_hits) {
                    return candidate.prefill_hits < current_best.prefill_hits;
                }
            }
        }

        // 3. Fall back to LRU (evict the oldest)
        return candidate.last_used_tick < current_best.last_used_tick;
    }

    static int64_t calculate_retention_score(
            const ExpertSlot & slot,
            uint32_t request_decode_count,
            uint32_t request_prefill_count,
            int current_layer,
            int total_layers) {
        int64_t score =
                int64_t(request_decode_count) * 1000000ll +
                int64_t(request_prefill_count) * 10000ll +
                int64_t(slot.decode_hits) * 10ll +
                int64_t(slot.prefill_hits);

        if (total_layers > 0 && slot.layer_idx >= 0) {
            const int normalized_layer = slot.layer_idx % total_layers;
            const int normalized_current = std::max(0, current_layer) % total_layers;
            const int distance = normalized_layer >= normalized_current
                    ? normalized_layer - normalized_current
                    : total_layers - normalized_current + normalized_layer;
            const int urgency = total_layers - distance;
            score += int64_t(urgency) * 1000ll;
            score += int64_t(total_layers - normalized_layer) * 100ll;
        }

        if (slot.prefetch_hits > 0 &&
                request_decode_count == 0 &&
                request_prefill_count == 0) {
            score -= 100000ll;
        }

        return score;
    }

    static int64_t calculate_hybrid_retention_score(
            const ExpertSlot & slot,
            uint32_t request_decode_count,
            uint32_t request_prefill_count,
            int current_layer,
            int total_layers,
            int layer_residents,
            int layer_target,
            int arc_target_recent,
            int arc_recent_residents,
            int contiguous_neighbors,
            uint64_t reload_bytes) {
        int64_t score = calculate_retention_score(
                slot,
                request_decode_count,
                request_prefill_count,
                current_layer,
                total_layers);

        if (slot.arc_segment == dstorage_arc_segment::frequent) {
            score += 1000000ll;
        } else if (slot.arc_segment == dstorage_arc_segment::recent) {
            score += 10000ll;
        }

        score += int64_t(std::max(0, contiguous_neighbors)) * 5000ll;
        score += int64_t(reload_bytes / 4096ull);

        if (layer_target > 0 && layer_residents > layer_target) {
            score -= int64_t(layer_residents - layer_target) * 2000ll;
        }

        const bool prefer_recent_victim = arc_recent_residents > arc_target_recent;
        const bool preferred_arc_victim =
                (prefer_recent_victim && slot.arc_segment == dstorage_arc_segment::recent) ||
                (!prefer_recent_victim && slot.arc_segment == dstorage_arc_segment::frequent);
        if (preferred_arc_victim) {
            score -= 25000ll;
        }

        return score;
    }

    static bool slot_is_protected(const ExpertSlot & slot) {
        return slot.execution_pins > 0 || slot.transfer_pending || slot.admission_pending;
    }

    static int calculate_speculative_slot_count(int n_slots, int n_expert_used) {
        if (n_slots < 64 || n_expert_used <= 0) {
            return 0;
        }
        const char * env = std::getenv("LLAMA_DSTORAGE_SPECULATIVE_SLOTS");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }

        const int requested = std::atoi(env);
        if (requested <= 0) {
            return 0;
        }
        return std::min(n_slots / 4, requested);
    }

    static bool slot_in_speculative_partition(int slot_idx, int n_slots, int speculative_slots) {
        return speculative_slots > 0 && slot_idx >= n_slots - speculative_slots;
    }

    static bool should_admit_prefetch(
            double confidence,
            int64_t expected_saved_stall,
            int64_t victim_retention_cost,
            bool has_free_slot,
            double min_confidence = 0.5) {
        if (confidence < min_confidence || expected_saved_stall <= 0) {
            return false;
        }
        return has_free_slot || expected_saved_stall > victim_retention_cost;
    }

    static int select_prefetch_distance(
            double transfer_us,
            double layer_compute_us,
            double confidence,
            double cache_pressure,
            double miss_rate,
            int max_distance) {
        if (confidence < 0.5 || transfer_us <= 0.0 || layer_compute_us <= 0.0 || max_distance <= 0) {
            return 0;
        }
        if (cache_pressure >= 0.9 && miss_rate < 0.2) {
            return 0;
        }
        const int distance = int(std::ceil(transfer_us / layer_compute_us));
        return std::max(1, std::min(distance, max_distance));
    }

    // Register a fused expert tensor (e.g., gate_up_exps or down_exps).
    // tensor_name: the GGML tensor name (e.g., "blk.5.ffn_gate_up_exps.weight")
    // file_path: wide path to the GGUF model file
    // file_offset: byte offset of the tensor data in the file
    // total_size: total byte size of the entire tensor (all experts)
    // n_experts: number of experts packed in this tensor
    void register_expert_tensor(
        int layer_idx,
        const char * tensor_name,
        const std::wstring & file_path,
        uint64_t file_offset,
        uint64_t total_size,
        int n_experts,
        uint64_t total_size_compressed = 0,
        uint64_t block_size = 0
    );

    // Called by the eval callback when the router has selected experts.
    // Handles ALL registered tensor types for the given layer in one call.
    // expert_ids: array of selected expert indices (length n_selected)
    // n_selected: number of selected experts
    // out_pool_ptrs: receives full_tensor_name -> CUDA pool base pointer for each type
    // out_id_map: receives the mapping from position i -> slot index
    // Returns true on success.
    bool ensure_experts_loaded(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected,
        std::unordered_map<std::string, uint64_t> & out_pool_ptrs,
        std::vector<int32_t> & out_id_map,
        dstorage_moe_phase phase = dstorage_moe_phase::decode
    );

    // Best-effort speculative cache warmup. This uses the same cache and streaming
    // path as the real router, but it does not return slot remaps for computation.
    bool prefetch_experts(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected
    );

    // Starts a best-effort prefetch on a background task. The actual router path
    // remains authoritative and will naturally wait if it reaches the same shared
    // cache while a speculative prefetch is still mutating it.
    bool prefetch_experts_async(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected
    );
    bool prefetch_experts_async_force(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected
    );
    bool enqueue_global_prefetch(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected
    );
    bool flush_global_prefetch_async();

    // Synchronous profile-driven VRAM warmup. Uses the normal decode load path
    // so loaded experts are treated as high-value decode cache entries.
    bool prewarm_vram_experts(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected
    );

    // Synchronously stages experts through VRAM in top-k-sized chunks and lets
    // the existing pinned-admission path copy them into the pinned host L2 cache.
    bool prewarm_pinned_experts(
        int layer_idx,
        const int32_t * expert_ids,
        int n_selected
    );

    // Block and wait for any in-flight speculative prefetch for the given layer.
    void wait_for_layer_prefetch(int layer_idx);
    void release_layer_experts(int layer_idx);
    void record_layer_compute_us(int64_t compute_us);
    void set_speculative_prefetch_enabled(bool enabled);
    int recommended_prefetch_distance(double confidence);
    double prefetch_confidence(int layer_idx, const int32_t * expert_ids, int n_selected);
    std::vector<int32_t> history_prefetch_candidates(int layer_idx, int max_experts);
    std::vector<int32_t> completion_prefetch_candidates(int layer_idx, int max_experts);

    // Get number of slots
    int get_n_slots() const { return n_slots_; }
    int get_active_slot_count() const { return active_slot_count_; }

    // Get actual slot stride (in bytes) allocated for a given tensor type pool
    uint64_t get_slot_stride(const std::string & tensor_name);
    int get_slot_count(const std::string & tensor_name);

    struct PinnedSlice {
        std::string tensor_name;
        uint64_t offset = 0;
        uint64_t size = 0;
    };

    struct PinnedEntry {
        int layer_idx = -1;
        int expert_idx = -1;
        void * host_ptr = nullptr;
        uint64_t bytes = 0;
        bool is_pinned = false;
        uint64_t last_used_tick = 0;
        std::vector<PinnedSlice> slices;
    };

    const PinnedEntry * get_pinned_entry(int layer_idx, int expert_idx);

    void destroy();
    ~DStorageSlotManager();

private:
    struct QueuedGlobalPrefetch {
        int layer_idx = -1;
        std::vector<int32_t> expert_ids;
    };

    static std::string make_tensor_key(int layer_idx, const char * tensor_name);
    static std::string get_tensor_type_key(const std::string & tensor_name);
    static int parse_layer_idx(const std::string & tensor_name);
    static std::string make_pool_key(int pool_group, const std::string & type_key);
    int pool_group_for_layer(int layer_idx) const;
    std::string pool_key_for_layer(int layer_idx, const std::string & tensor_name) const;
    bool slot_compatible_with_layer(int slot_idx, int layer_idx) const;
    bool ensure_pools_allocated();
    static uint64_t make_expert_key(int layer_idx, int expert_idx);
    uint32_t pinned_admit_min_hits() const;
    bool pinned_cache_enabled() const;
    bool pinned_admission_pending(uint64_t expert_key) const;
    bool static_pinned_entry(uint64_t expert_key) const;
    bool preload_static_pinned_layers();
    void release_prefill_only_pinned_cache();
    bool prefetch_global_batch(std::vector<QueuedGlobalPrefetch> batch);
    void collect_async_prefetches(bool wait_all);
    void collect_pinned_admissions(bool wait_all);
    void destroy_pinned_cache();
    bool ensure_active_pools_allocated(int active_capacity);
    bool hybrid_arc_enabled() const;
    int layer_target_slots(int layer_idx) const;
    uint64_t expert_reload_bytes(int layer_idx) const;
    int contiguous_resident_neighbors(const ExpertSlot & slot) const;
    void reset_arc_state();
    void arc_record_hit(ExpertSlot & slot, dstorage_moe_phase phase);
    void arc_record_eviction(const ExpertSlot & slot);
    void arc_record_admission(ExpertSlot & slot, dstorage_moe_phase phase);
    uint64_t decode_hot_score(uint64_t expert_key) const;
    bool decode_hot_protected(uint64_t expert_key) const;
    bool decode_hot_protected(const ExpertSlot & slot) const;
    void rebuild_decode_hot_protected_set();
    void write_routing_trace(
            int layer_idx,
            dstorage_moe_phase phase,
            const std::vector<int32_t> & experts,
            const std::vector<int32_t> & misses,
            uint64_t stream_file_bytes,
            uint64_t transfer_us,
            uint64_t total_us);
    bool is_better_request_eviction_candidate(
            const ExpertSlot & candidate,
            const ExpertSlot & current_best,
            int current_layer) const;
    int speculative_slot_count() const;

    // Per-type VRAM pool: one allocation per tensor type (e.g. ffn_gate_up_exps.weight)
    struct TensorTypePool {
        uint64_t cuda_ptr  = 0;
        uint64_t alloc_ptr = 0;
        uint64_t slot_size = 0;
        bool owns_alloc = true;
    };


    struct PinnedReadOp {
        std::wstring file_path;
        uint64_t file_offset = 0;
        uint64_t dst_offset = 0;
        uint64_t size = 0;
    };

    struct PinnedAdmissionResult {
        uint64_t expert_key = 0;
        int slot_idx = -1;
        bool ok = false;
        uint64_t admit_us = 0;
        PinnedEntry entry;
    };

    struct PinnedAdmissionWork {
        PinnedAdmissionResult result;
        std::vector<PinnedReadOp> read_ops;
    };

    struct PendingPinnedAdmission {
        std::vector<uint64_t> expert_keys;
        std::vector<int> slot_indices;
        uint64_t bytes = 0;
        std::future<std::vector<PinnedAdmissionResult>> future;
    };

    struct PendingAsyncPrefetch {
        int layer_idx = -1;
        int n_selected = 0;
        std::vector<int> affected_layers;
        std::future<bool> future;
    };

    DSLoaderHandle ds_loader_ = nullptr;
    std::mutex slots_mutex_;
    std::mutex async_prefetch_mutex_;

    int n_slots_ = 0;
    uint64_t pool_budget_bytes_ = 0;
    uint64_t pool_budget_bytes_prefill_ = 0;
    uint64_t pool_budget_bytes_decode_ = 0;
    dstorage_moe_phase current_allocated_phase_ = dstorage_moe_phase::prefill;
    bool pools_allocated_ = false;
    bool phase_cache_policy_ = false;
    bool hybrid_arc_policy_ = false;
    bool speculative_prefetch_enabled_ = false;
    uint64_t use_tick_ = 0;
    uint64_t pinned_budget_bytes_ = 0;
    uint64_t pinned_used_bytes_ = 0;
    uint64_t pinned_pending_bytes_ = 0;
    uint64_t pinned_tick_ = 0;
    FILE * routing_trace_file_ = nullptr;
    uint64_t routing_trace_sequence_ = 0;

    // Slot occupancy tracking — shared across all tensor types.
    // Slot i in every TensorTypePool always holds the same expert.
    std::vector<ExpertSlot> slots_;
    std::vector<int> pool_group_slot_counts_;

    // Per-type VRAM pools keyed by type key (e.g. "ffn_gate_up_exps.weight")
    std::unordered_map<std::string, TensorTypePool> type_pools_;
    std::unordered_map<std::string, TensorTypePool> active_type_pools_;
    std::unordered_map<std::string, uint64_t> type_strides_;
    std::unordered_map<int, int> layer_pool_groups_;
    std::vector<uint64_t> layer_reload_bytes_;
    int active_capacity_ = 0;
    int active_slot_count_ = 0;
    uint64_t bundle_staging_ptr_ = 0;
    uint64_t bundle_staging_size_ = 0;

    // Registry of expert tensors (key: "layer_idx:tensor_name")
    std::unordered_map<std::string, ExpertTensorInfo> tensor_registry_;

    // Per-layer list of registered tensor names
    std::unordered_map<int, std::vector<std::string>> layer_tensors_;
    std::unordered_map<uint64_t, PinnedEntry> pinned_cache_;
    std::unordered_set<uint64_t> static_pinned_experts_;
    std::vector<int> static_pinned_layers_;
    std::vector<std::pair<int, int>> static_pinned_expert_selection_;
    bool static_pinned_preload_done_ = false;
    bool prefill_only_pinned_released_ = false;
    std::unordered_map<uint64_t, uint32_t> pinned_frequency_;
    std::unordered_map<uint64_t, int> resident_experts_;
    std::vector<PendingPinnedAdmission> pinned_admissions_;
    std::vector<PendingAsyncPrefetch> async_prefetches_;
    std::vector<QueuedGlobalPrefetch> global_prefetch_queue_;
    std::unordered_map<int, std::vector<int>> execution_slots_by_layer_;

    struct LayerArcState {
        int target_recent = 0;
        int recent_residents = 0;
        int frequent_residents = 0;
        std::deque<int> recent_ghost;
        std::deque<int> frequent_ghost;
        std::unordered_set<int> recent_ghost_set;
        std::unordered_set<int> frequent_ghost_set;
    };

    std::vector<LayerArcState> layer_arc_;

    struct PhaseStats {
        uint64_t calls = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t all_hit_calls = 0;
        uint64_t miss_calls = 0;
        uint64_t evicted_prefill_only = 0;
        uint64_t evicted_decode_touched = 0;
        uint64_t stream_calls = 0;
        uint64_t stream_cold_experts = 0;
        uint64_t stream_pinned_experts = 0;
        uint64_t stream_possible_runs = 0;
        uint64_t stream_possible_run_experts = 0;
        uint64_t stream_runs = 0;
        uint64_t stream_run_experts = 0;
        uint64_t stream_requests = 0;
        uint64_t stream_chunks = 0;
        uint64_t stream_file_bytes = 0;
        uint64_t stream_cuda_bytes = 0;
        uint64_t stream_pinned_bytes = 0;
        uint64_t stream_pinned_us = 0;
        uint64_t stream_staging_bytes = 0;
        uint64_t stream_submit_us = 0;
        uint64_t stream_wait_us = 0;
        uint64_t stream_total_us = 0;
        uint64_t stream_max_requests_per_call = 0;
        uint64_t stream_max_chunks_per_call = 0;
        uint64_t stream_aligned_4k_requests = 0;
        uint64_t stream_unaligned_4k_requests = 0;
        uint64_t stream_file_offset_4k_aligned = 0;
        uint64_t stream_size_4k_aligned = 0;
        uint64_t stream_cuda_dest_4k_aligned = 0;
        uint64_t stream_uncompressed_4k_aligned = 0;
        uint64_t stream_file_offset_mod_min = UINT64_MAX;
        uint64_t stream_file_offset_mod_max = 0;
        uint64_t stream_file_offset_mod_gcd = 0;
        uint64_t stream_size_mod_min = UINT64_MAX;
        uint64_t stream_size_mod_max = 0;
        uint64_t stream_size_mod_gcd = 0;
        uint64_t stream_cuda_dest_mod_min = UINT64_MAX;
        uint64_t stream_cuda_dest_mod_max = 0;
        uint64_t stream_cuda_dest_mod_gcd = 0;
        uint64_t stream_uncompressed_mod_min = UINT64_MAX;
        uint64_t stream_uncompressed_mod_max = 0;
        uint64_t stream_uncompressed_mod_gcd = 0;
        uint64_t stream_contiguous_request_pairs = 0;
        uint64_t stream_forward_gap_bytes = 0;
        uint64_t stream_backward_or_overlap_pairs = 0;
        uint64_t stream_size_le_64k = 0;
        uint64_t stream_size_le_256k = 0;
        uint64_t stream_size_le_1m = 0;
        uint64_t stream_size_le_4m = 0;
        uint64_t stream_size_gt_4m = 0;
        uint64_t stream_min_request_bytes = UINT64_MAX;
        uint64_t stream_max_request_bytes = 0;
        uint64_t stream_max_chunk_staging_bytes = 0;
    };

    struct DiagnosticStats {
        uint64_t prefetch_candidate_calls = 0;
        uint64_t prefetch_reject_observations = 0;
        uint64_t prefetch_reject_resident = 0;
        uint64_t prefetch_reject_policy = 0;
        uint64_t prefetch_submitted_calls = 0;
        uint64_t prefetch_submitted_experts = 0;
        uint64_t prefetch_completed_ok = 0;
        uint64_t prefetch_completed_failed = 0;
        uint64_t prefetch_wait_calls = 0;
        uint64_t prefetch_wait_us = 0;
        uint64_t prefetch_loaded_experts = 0;
        uint64_t useful_prefetch_hits = 0;
        uint64_t useful_prefetch_prefill_hits = 0;
        uint64_t useful_prefetch_decode_hits = 0;
        uint64_t prefetch_evicted_before_use = 0;
        uint64_t prefetch_evicted_after_use = 0;
        double prefetch_best_confidence = 0.0;
        int64_t prefetch_best_candidate_value = 0;
        int64_t prefetch_lowest_victim_cost = std::numeric_limits<int64_t>::max();
        uint64_t pinned_admit_queued = 0;
        uint64_t pinned_admit_completed = 0;
        uint64_t pinned_admit_dropped = 0;
        uint64_t pinned_admit_bytes = 0;
        uint64_t pinned_admit_us = 0;
        uint64_t pinned_admit_cuda_pinned = 0;
        uint64_t pinned_admit_fallback_ram = 0;
        uint64_t pinned_admit_skipped_policy = 0;
        uint64_t pinned_admit_skipped_pending = 0;
        uint64_t pinned_admit_skipped_budget = 0;
        uint64_t pinned_admit_skipped_throttle = 0;
        uint64_t pinned_prefill_only_releases = 0;
        uint64_t pinned_prefill_only_release_entries = 0;
        uint64_t pinned_prefill_only_release_bytes = 0;
        uint64_t pinned_prefill_only_release_us = 0;
    };

    struct DecodeHotStats {
        uint64_t accesses = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t protected_hits = 0;
        uint64_t protected_misses = 0;
        uint64_t protected_evictions = 0;
        uint64_t rebuilds = 0;
    };

    struct TimelineStats {
        uint64_t real_calls = 0;
        uint64_t prefill_calls = 0;
        uint64_t decode_calls = 0;
        uint64_t all_hit_calls = 0;
        uint64_t miss_calls = 0;
        uint64_t selected_experts = 0;
        uint64_t unique_experts = 0;
        uint64_t hit_experts = 0;
        uint64_t miss_experts = 0;
        uint64_t ensure_total_us = 0;
        uint64_t stream_total_us = 0;
        uint64_t stream_file_bytes = 0;
        uint64_t compute_samples = 0;
        uint64_t compute_total_us = 0;
        uint64_t compute_min_us = std::numeric_limits<uint64_t>::max();
        uint64_t compute_max_us = 0;
    };

    struct SchedulerDryRunStats {
        uint64_t miss_calls = 0;
        uint64_t no_window_calls = 0;
        uint64_t full_hide_calls = 0;
        uint64_t partial_hide_calls = 0;
        uint64_t transfer_total_us = 0;
        uint64_t prior_compute_total_us = 0;
        uint64_t hidden_possible_us = 0;
        uint64_t visible_after_overlap_us = 0;
        uint64_t file_bytes = 0;
        uint64_t hideable_file_bytes = 0;
        int64_t best_margin_us = std::numeric_limits<int64_t>::min();
        int64_t worst_margin_us = std::numeric_limits<int64_t>::max();
    };

    PhaseStats prefill_stats_;
    PhaseStats decode_stats_;
    PhaseStats prefetch_stats_;
    DiagnosticStats diagnostic_stats_;
    DecodeHotStats decode_hot_stats_;
    TimelineStats timeline_stats_;
    SchedulerDryRunStats scheduler_dry_run_stats_;
    SchedulerDryRunStats scheduler_prefill_dry_run_stats_;
    SchedulerDryRunStats scheduler_decode_dry_run_stats_;
    RequestActivationStats request_activation_stats_;
    dstorage_moe_phase last_request_phase_ = dstorage_moe_phase::prefill;
    bool request_tracking_started_ = false;
    double transfer_ewma_us_ = 0.0;
    double layer_compute_ewma_us_ = 0.0;
    uint64_t last_layer_compute_us_ = 0;
    std::unordered_map<uint64_t, uint32_t> decode_hot_counts_;
    std::unordered_map<uint64_t, uint32_t> decode_hot_misses_;
    std::unordered_map<uint64_t, uint64_t> decode_hot_last_seen_;
    std::unordered_set<uint64_t> decode_hot_protected_;
    uint64_t decode_hot_tick_ = 0;

    int n_layers_      = 0;
    int n_experts_     = 0;
    int n_expert_used_ = 0;
};
