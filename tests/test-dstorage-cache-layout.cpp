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
            0) && ok;
    ok = expect_equal(
            "prefill_token_chunk_rejects_insufficient_topk",
            uint64_t(DStorageSlotManager::calculate_prefill_token_chunk(27, 26)),
            0) && ok;
    ok = expect_equal(
            "prefill_token_chunk_exact_topk_capacity",
            uint64_t(DStorageSlotManager::calculate_prefill_token_chunk(27, 27)),
            1) && ok;
    ok = expect_equal(
            "prefill_minimum_pool_group_capacity",
            uint64_t(DStorageSlotManager::calculate_prefill_pool_group_capacity({ 400, 240 }, 0)),
            240) && ok;
    ok = expect_equal(
            "prefill_minimum_pool_group_capacity_after_speculative_reserve",
            uint64_t(DStorageSlotManager::calculate_prefill_pool_group_capacity({ 400, 240 }, 64)),
            216) && ok;
    ok = expect_equal(
            "prefill_pool_group_reserve_rejects_topk_starvation",
            uint64_t(DStorageSlotManager::calculate_prefill_token_chunk(
                    27,
                    DStorageSlotManager::calculate_prefill_pool_group_capacity({ 100, 27 }, 64))),
            0) && ok;
    ok = expect_equal(
            "prefill_preload_possible_speculative_reserve",
            uint64_t(DStorageSlotManager::calculate_prefill_speculative_reserve(
                    640, 27, false, true, false, false, 0)),
            160) && ok;
    ok = expect_equal(
            "speculative_reserve_always_explicit_slots",
            uint64_t(DStorageSlotManager::calculate_prefill_speculative_reserve(
                    640, 27, false, false, true, true, 48)),
            48) && ok;
    ok = expect_equal(
            "speculative_reserve_always_without_explicit_slots",
            uint64_t(DStorageSlotManager::calculate_prefill_speculative_reserve(
                    640, 27, false, true, true, false, 0)),
            0) && ok;
    ok = expect_equal(
            "explicit_speculative_slots_without_policy_are_ignored",
            uint64_t(DStorageSlotManager::calculate_prefill_speculative_reserve(
                    640, 27, false, false, false, true, 48)),
            0) && ok;
    ok = expect_equal(
            "max_prefill_workspace_experts",
            uint64_t(DStorageSlotManager::max_prefill_workspace_experts()),
            192) && ok;
    ok = expect_equal(
            "speculative_slots_default_off",
            uint64_t(DStorageSlotManager::calculate_speculative_slot_count(1039, 8)),
            0) && ok;
    ok = expect_equal(
            "speculative_reserve_disabled_by_default",
            DStorageSlotManager::should_reserve_speculative(false, false) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "speculative_reserve_enabled",
            DStorageSlotManager::should_reserve_speculative(true, false) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "speculative_reserve_always_overrides_temporary_disable",
            DStorageSlotManager::should_reserve_speculative(false, true) ? 1 : 0,
            1) && ok;
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

    // Next-use eviction: with equal use rates the layer that runs next is needed soonest and the layer that just ran
    // is needed a full pass later; rarer experts are expected later still, never-seen ones last.
    {
        const double decay = 0.75;
        const double every_pass = 1.0 / (1.0 - decay);
        const double upcoming = DStorageSlotManager::calculate_expected_passes_to_next_use(every_pass, decay, 11, 10, 41);
        const double just_ran = DStorageSlotManager::calculate_expected_passes_to_next_use(every_pass, decay, 9, 10, 41);
        const double current = DStorageSlotManager::calculate_expected_passes_to_next_use(every_pass, decay, 10, 10, 41);
        const double half_rate = DStorageSlotManager::calculate_expected_passes_to_next_use(0.5 * every_pass, decay, 11, 10, 41);
        const double never = DStorageSlotManager::calculate_expected_passes_to_next_use(0.0, decay, 11, 10, 41);
        ok = expect_equal("next_use_upcoming_before_just_ran", upcoming < just_ran ? 1 : 0, 1) && ok;
        ok = expect_equal("next_use_current_layer_one_pass", current == 1.0 ? 1 : 0, 1) && ok;
        ok = expect_equal("next_use_rate_dominates_layer", just_ran < half_rate ? 1 : 0, 1) && ok;
        ok = expect_equal("next_use_never_seen_last", half_rate < never ? 1 : 0, 1) && ok;
    }

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
    protected_slot.admission_pending = false;
    protected_slot.transfer_quarantined = true;
    ok = expect_equal(
            "quarantined_transfer_protects",
            DStorageSlotManager::slot_is_protected(protected_slot) ? 1 : 0,
            1) && ok;

    const int32_t learned_ids[] = { 2, 4 };
    const int32_t invalid_learned_ids[] = { 2, 16 };
    ok = expect_equal(
            "learned_request_valid",
            DStorageSlotManager::learned_prefetch_request_valid(learned_ids, 2, 16, true) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "learned_request_reject_invalid_id",
            DStorageSlotManager::learned_prefetch_request_valid(invalid_learned_ids, 2, 16, true) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "learned_request_reject_cancelled",
            DStorageSlotManager::learned_prefetch_request_valid(learned_ids, 2, 16, false) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "learned_request_max_four",
            uint64_t(DStorageSlotManager::max_learned_prefetch_experts()),
            4) && ok;

    std::vector<ExpertSlot> learned_slots(3);
    std::vector<uint8_t> speculative_compatible = { 1, 1, 0 };
    ok = expect_equal(
            "learned_free_speculative_capacity",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    2, learned_slots, speculative_compatible) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "learned_free_capacity_count",
            DStorageSlotManager::learned_prefetch_available_slots(
                    learned_slots, speculative_compatible),
            2) && ok;
    ok = expect_equal(
            "learned_reject_demand_partition",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    3, learned_slots, speculative_compatible) ? 1 : 0,
            0) && ok;
    learned_slots[0].execution_pins = 1;
    ok = expect_equal(
            "learned_capacity_skips_execution_pin",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    2, learned_slots, speculative_compatible) ? 1 : 0,
            0) && ok;
    learned_slots[0].execution_pins = 0;
    learned_slots[1].transfer_pending = true;
    ok = expect_equal(
            "learned_capacity_skips_transfer_pending",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    2, learned_slots, speculative_compatible) ? 1 : 0,
            0) && ok;
    learned_slots[1].transfer_pending = false;
    learned_slots[1].admission_pending = true;
    ok = expect_equal(
            "learned_capacity_skips_admission_pending",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    2, learned_slots, speculative_compatible) ? 1 : 0,
            0) && ok;
    learned_slots[1].admission_pending = false;
    learned_slots[1].transfer_quarantined = true;
    ok = expect_equal(
            "learned_capacity_skips_quarantined_transfer",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    2, learned_slots, speculative_compatible) ? 1 : 0,
            0) && ok;
    learned_slots[1].transfer_quarantined = false;
    learned_slots[0].occupied = true;
    learned_slots[1].occupied = true;
    ok = expect_equal(
            "learned_reject_full_cache",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, learned_slots, speculative_compatible) ? 1 : 0,
            0) && ok;

    ExpertSlot consumed_learned;
    consumed_learned.layer_idx = 4;
    consumed_learned.occupied = true;
    consumed_learned.learned_prefetch = true;
    consumed_learned.learned_prefetch_used = true;
    consumed_learned.last_used_tick = 10;
    std::vector<ExpertSlot> recyclable_slots = { consumed_learned };
    std::vector<uint8_t> recyclable_compatible = { 1 };
    ok = expect_equal(
            "learned_recycle_consumed_unpinned",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "learned_recycled_capacity_count",
            DStorageSlotManager::learned_prefetch_available_slots(
                    recyclable_slots, recyclable_compatible, 12),
            1) && ok;

    recyclable_slots[0].learned_prefetch_used = false;
    ok = expect_equal(
            "learned_recycle_reject_unused",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0] = consumed_learned;
    recyclable_slots[0].learned_prefetch = false;
    ok = expect_equal(
            "learned_recycle_reject_ordinary_entry",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0] = consumed_learned;
    recyclable_slots[0].execution_pins = 1;
    ok = expect_equal(
            "learned_recycle_reject_execution_pin",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0] = consumed_learned;
    recyclable_slots[0].transfer_pending = true;
    ok = expect_equal(
            "learned_recycle_reject_transfer_pending",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0] = consumed_learned;
    recyclable_slots[0].transfer_quarantined = true;
    ok = expect_equal(
            "learned_recycle_reject_quarantined",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0] = consumed_learned;
    recyclable_slots[0].admission_pending = true;
    ok = expect_equal(
            "learned_recycle_reject_admission_pending",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0] = consumed_learned;
    ok = expect_equal(
            "learned_recycle_reject_incompatible",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, std::vector<uint8_t>{ 0 }, 12) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "learned_recycle_reject_demand_partition",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1,
                    recyclable_slots,
                    std::vector<uint8_t>{ uint8_t(DStorageSlotManager::slot_in_speculative_partition(0, 2, 1)) },
                    12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0].layer_idx = 12;
    ok = expect_equal(
            "learned_recycle_reject_target_layer_guard",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;
    recyclable_slots[0].layer_idx = 11;
    ok = expect_equal(
            "learned_recycle_reject_previous_layer_guard",
            DStorageSlotManager::learned_prefetch_has_capacity(
                    1, recyclable_slots, recyclable_compatible, 12) ? 1 : 0,
            0) && ok;

    DStorageSlotManager uninitialized_manager;
    const auto cancelled_token = std::make_shared<std::atomic<bool>>(false);
    const auto live_token = std::make_shared<std::atomic<bool>>(true);
    ok = expect_equal(
            "learned_api_rejects_null_token_without_gpu",
            uninitialized_manager.prefetch_learned_experts_async(0, learned_ids, 2, nullptr) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "learned_api_rejects_cancelled_token_without_gpu",
            uninitialized_manager.prefetch_learned_experts_async(0, learned_ids, 2, cancelled_token) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "learned_api_rejects_uninitialized_manager_without_gpu",
            uninitialized_manager.prefetch_learned_experts_async(0, learned_ids, 2, live_token) ? 1 : 0,
            0) && ok;
    ok = expect_equal(
            "learned_shadow_count_uninitialized",
            uninitialized_manager.count_learned_nonresident(0, learned_ids, 2) == -1 ? 1 : 0,
            1) && ok;
    ok = expect_equal(
            "learned_available_slots_uninitialized",
            uninitialized_manager.learned_prefetch_available_slots(0),
            0) && ok;
    const auto learned_snapshot = uninitialized_manager.telemetry_snapshot();
    ok = expect_equal("learned_rejected_snapshot", learned_snapshot.learned_prefetch_rejected, 3) && ok;
    ok = expect_equal("learned_cancelled_snapshot", learned_snapshot.learned_prefetch_cancelled, 1) && ok;

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
