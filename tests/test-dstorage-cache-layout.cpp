#include "llama-dstorage-slots.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>

static bool expect_equal(const char * name, uint64_t actual, uint64_t expected) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(
            stderr,
            "expected %s=%llu, got %llu\n",
            name,
            (unsigned long long) expected,
            (unsigned long long) actual);
    return false;
}

static void set_test_env(const char * name, const char * value) {
#if defined(_WIN32)
    const std::string assignment = std::string(name) + "=" + value;
    _putenv(assignment.c_str());
#else
    setenv(name, value, 1);
#endif
}

int main() {
    constexpr uint64_t mib = 1024ull * 1024ull;
    constexpr uint64_t requested_cache = 2048ull * mib;
    constexpr uint64_t bytes_per_slot = 2ull * mib;
    constexpr int n_layers = 40;
    constexpr int n_experts = 256;

    const auto layout = DStorageSlotManager::calculate_cache_layout(
            requested_cache,
            bytes_per_slot,
            n_layers,
            n_experts);

    bool ok = true;
    ok = expect_equal(
            "persistent_cache_budget_bytes",
            layout.persistent_cache_budget_bytes,
            requested_cache) && ok;
    ok = expect_equal("workspace_reserve_bytes", layout.workspace_reserve_bytes, 0) && ok;
    ok = expect_equal("n_persistent_slots", uint64_t(layout.n_persistent_slots), 1024) && ok;
    ok = expect_equal(
            "phase_policy_disabled_prefill_mib",
            DStorageSlotManager::calculate_phase_cache_mib(2048, false, true),
            2048) && ok;
    ok = expect_equal(
            "phase_policy_enabled_prefill_mib",
            DStorageSlotManager::calculate_phase_cache_mib(2048, true, true),
            1000) && ok;
    ok = expect_equal(
            "phase_policy_enabled_decode_mib",
            DStorageSlotManager::calculate_phase_cache_mib(2048, true, false),
            2048) && ok;
    ok = expect_equal(
            "phase_policy_disabled_no_reallocation",
            DStorageSlotManager::should_reallocate_for_phase(
                    false,
                    dstorage_moe_phase::prefill,
                    dstorage_moe_phase::decode,
                    2048,
                    2048) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "different_phase_budgets_reallocate",
            DStorageSlotManager::should_reallocate_for_phase(
                    true,
                    dstorage_moe_phase::prefill,
                    dstorage_moe_phase::decode,
                    1000,
                    2048) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "decode_workspace_capacity",
            uint64_t(DStorageSlotManager::calculate_decode_workspace_capacity(8, 8)),
            8) && ok;
    ok = expect_equal(
            "decode_workspace_capacity_overflow",
            uint64_t(DStorageSlotManager::calculate_decode_workspace_capacity(9, 8)),
            0) && ok;
    ok = expect_equal(
            "prefill_token_chunk",
            uint64_t(DStorageSlotManager::calculate_prefill_token_chunk(8, 64)),
            8) && ok;
    ok = expect_equal(
            "prefill_token_chunk_large_topk",
            uint64_t(DStorageSlotManager::calculate_prefill_token_chunk(128, 64)),
            1) && ok;
    ok = expect_equal(
            "max_prefill_workspace_experts",
            uint64_t(DStorageSlotManager::max_prefill_workspace_experts()),
            192) && ok;
    ok = expect_equal(
            "speculative_slots_default_off",
            uint64_t(DStorageSlotManager::calculate_speculative_slot_count(1039, 8)),
            0) && ok;
    set_test_env("LLAMA_DSTORAGE_SPECULATIVE_SLOTS", "16");
    ok = expect_equal(
            "speculative_slots_env",
            uint64_t(DStorageSlotManager::calculate_speculative_slot_count(1039, 8)),
            16) && ok;
    set_test_env("LLAMA_DSTORAGE_SPECULATIVE_SLOTS", "");
    ok = expect_equal(
            "speculative_partition_tail",
            DStorageSlotManager::slot_in_speculative_partition(1038, 1039, 16) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "demand_partition_head",
            DStorageSlotManager::slot_in_speculative_partition(1000, 1039, 16) ? 1 : 0,
            0) && ok;

    DStorageSlotManager::RequestActivationStats request_stats;
    const int32_t prefill_ids[] = { 3, 3, 7 };
    const int32_t decode_ids[] = { 3, 9 };
    request_stats.begin_request();
    request_stats.record(2, prefill_ids, 3, dstorage_moe_phase::prefill);
    request_stats.record(2, decode_ids, 2, dstorage_moe_phase::decode);
    ok = expect_equal(
            "request_prefill_count",
            request_stats.count(2, 3, dstorage_moe_phase::prefill),
            2) && ok;
    ok = expect_equal(
            "request_decode_count",
            request_stats.count(2, 3, dstorage_moe_phase::decode),
            1) && ok;
    ok = expect_equal(
            "request_phase_isolation",
            request_stats.count(2, 7, dstorage_moe_phase::decode),
            0) && ok;
    request_stats.begin_request();
    ok = expect_equal(
            "request_reset",
            request_stats.count(2, 3, dstorage_moe_phase::prefill),
            0) && ok;
    ok = expect_equal("request_generation", request_stats.generation(), 2) && ok;
    request_stats.record(4, decode_ids, 2, dstorage_moe_phase::decode);
    request_stats.record(4, decode_ids, 2, dstorage_moe_phase::decode);
    const auto top_decode = request_stats.top_decode_experts(4, 2, 0.5);
    ok = expect_equal("top_decode_count", top_decode.size(), 2) && ok;
    ok = expect_equal("top_decode_first", top_decode.empty() ? 0 : uint64_t(top_decode[0]), 3) && ok;

    ExpertSlot reused;
    reused.layer_idx = 20;
    reused.decode_hits = 1;
    ExpertSlot stale;
    stale.layer_idx = 20;
    stale.decode_hits = 100000;
    const int64_t reused_score = DStorageSlotManager::calculate_retention_score(
            reused, 4, 0, 20, 40);
    const int64_t stale_score = DStorageSlotManager::calculate_retention_score(
            stale, 0, 0, 20, 40);
    ok = expect_equal("request_reuse_dominates", reused_score > stale_score ? 1 : 0, 1) && ok;

    ExpertSlot near_layer;
    near_layer.layer_idx = 21;
    ExpertSlot far_layer;
    far_layer.layer_idx = 35;
    const int64_t near_score = DStorageSlotManager::calculate_retention_score(
            near_layer, 0, 0, 20, 40);
    const int64_t far_score = DStorageSlotManager::calculate_retention_score(
            far_layer, 0, 0, 20, 40);
    ok = expect_equal("layer_urgency", near_score > far_score ? 1 : 0, 1) && ok;

    ExpertSlot polluted;
    polluted.layer_idx = 21;
    polluted.prefetch_hits = 1000;
    const int64_t polluted_score = DStorageSlotManager::calculate_retention_score(
            polluted, 0, 0, 20, 40);
    ok = expect_equal("prefetch_pollution_penalty", polluted_score < near_score ? 1 : 0, 1) && ok;

    ExpertSlot arc_recent;
    arc_recent.layer_idx = 10;
    arc_recent.arc_segment = dstorage_arc_segment::recent;
    ExpertSlot arc_frequent = arc_recent;
    arc_frequent.arc_segment = dstorage_arc_segment::frequent;
    const int64_t arc_recent_score = DStorageSlotManager::calculate_hybrid_retention_score(
            arc_recent, 0, 0, 10, 40, 30, 25, 12, 20, 0, 2 * mib);
    const int64_t arc_frequent_score = DStorageSlotManager::calculate_hybrid_retention_score(
            arc_frequent, 0, 0, 10, 40, 30, 25, 12, 20, 0, 2 * mib);
    ok = expect_equal(
            "arc_prefers_frequent_resident",
            arc_frequent_score > arc_recent_score ? 1 : 0,
            1) && ok;

    const int64_t contiguous_score = DStorageSlotManager::calculate_hybrid_retention_score(
            arc_recent, 0, 0, 10, 40, 25, 25, 12, 12, 2, 2 * mib);
    const int64_t isolated_score = DStorageSlotManager::calculate_hybrid_retention_score(
            arc_recent, 0, 0, 10, 40, 25, 25, 12, 12, 0, 2 * mib);
    ok = expect_equal(
            "contiguous_resident_value",
            contiguous_score > isolated_score ? 1 : 0,
            1) && ok;

    ExpertSlot shallow;
    shallow.layer_idx = 2;
    shallow.arc_segment = dstorage_arc_segment::recent;
    ExpertSlot deep = shallow;
    deep.layer_idx = 35;
    const int64_t shallow_hybrid_score = DStorageSlotManager::calculate_hybrid_retention_score(
            shallow, 0, 0, 0, 40, 20, 25, 12, 12, 0, 2 * mib);
    const int64_t deep_hybrid_score = DStorageSlotManager::calculate_hybrid_retention_score(
            deep, 0, 0, 0, 40, 20, 25, 12, 12, 0, 2 * mib);
    ok = expect_equal(
            "hybrid_shallow_protection",
            shallow_hybrid_score > deep_hybrid_score ? 1 : 0,
            1) && ok;

    ExpertSlot protected_slot;
    protected_slot.execution_pins = 1;
    ok = expect_equal(
            "execution_pin_protects",
            DStorageSlotManager::slot_is_protected(protected_slot) ? 1 : 0,
            1) && ok;
    protected_slot.execution_pins = 0;
    protected_slot.transfer_pending = true;
    ok = expect_equal(
            "transfer_protects",
            DStorageSlotManager::slot_is_protected(protected_slot) ? 1 : 0,
            1) && ok;
    protected_slot.transfer_pending = false;
    protected_slot.admission_pending = true;
    ok = expect_equal(
            "admission_protects",
            DStorageSlotManager::slot_is_protected(protected_slot) ? 1 : 0,
            1) && ok;

    ok = expect_equal(
            "reject_low_confidence_prefetch",
            DStorageSlotManager::should_admit_prefetch(0.25, 1000, 0, true) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "admit_relaxed_confidence_prefetch",
            DStorageSlotManager::should_admit_prefetch(0.25, 1000, 0, true, 0.25) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "admit_runtime_confidence_prefetch",
            DStorageSlotManager::should_admit_prefetch(0.125, 1000, 0, true, 0.125) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "reject_low_benefit_prefetch",
            DStorageSlotManager::should_admit_prefetch(0.75, 800, 900, false) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "admit_without_double_confidence_penalty",
            DStorageSlotManager::should_admit_prefetch(0.25, 1000, 900, false, 0.125) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "admit_high_value_prefetch",
            DStorageSlotManager::should_admit_prefetch(0.75, 2000, 900, false) ? 1 : 0,
            1) && ok;

    ok = expect_equal(
            "adaptive_distance_two",
            DStorageSlotManager::select_prefetch_distance(8000, 4000, 0.8, 0.5, 0.5, 4),
            2) && ok;
    ok = expect_equal(
            "adaptive_distance_three",
            DStorageSlotManager::select_prefetch_distance(9000, 3000, 0.8, 0.5, 0.5, 4),
            3) && ok;
    ok = expect_equal(
            "adaptive_reject_low_confidence",
            DStorageSlotManager::select_prefetch_distance(8000, 4000, 0.4, 0.5, 0.5, 4),
            0) && ok;
    ok = expect_equal(
            "adaptive_reject_cache_pollution",
            DStorageSlotManager::select_prefetch_distance(8000, 4000, 0.8, 0.95, 0.1, 4),
            0) && ok;

    std::printf(
            "DSTORAGE_CACHE_LAYOUT requested_mib=%llu persistent_mib=%llu workspace_mib=%llu slots=%d\n",
            (unsigned long long) (requested_cache / mib),
            (unsigned long long) (layout.persistent_cache_budget_bytes / mib),
            (unsigned long long) (layout.workspace_reserve_bytes / mib),
            layout.n_persistent_slots);

    return ok ? 0 : 1;
}
