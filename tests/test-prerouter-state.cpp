#include "../src/llama-prerouter-state.h"

#include <cstdio>
#include <initializer_list>
#include <utility>
#include <vector>

namespace {

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

std::vector<int32_t> ids(std::initializer_list<int32_t> values) {
    return std::vector<int32_t>(values);
}

bool test_batched_alignment_and_interleaving() {
    llama_prerouter_state state;
    if (!check(state.begin_batch({ { 0, 0 }, { 1, 0 }, { 0, 1 } }), "interleaved batch accepted")) {
        return false;
    }
    if (!check(state.route(0, 0, 6, ids({ 3, 5 })), "route stored for seq 0")) return false;
    if (!check(state.route(1, 0, 6, ids({ 9 })), "route stored for seq 1")) return false;
    if (!check(state.route(0, 1, 7, ids({ 4, 8 })), "target route stored")) return false;
    if (!check(state.previous_route(0, 1, 6) == ids({ 3, 5 }), "previous route uses pos-1 and owner layer")) return false;
    if (!check(state.predict(0, 0, 6, ids({ 4, 8 })), "prediction staged for owner+1")) return false;
    const auto result = state.observe(0, 1, 7, ids({ 4, 8 }));
    return check(result.has_value(), "aligned observation found") &&
        check(result->matched, "aligned prediction matched") &&
        check(result->hits == 2 && result->total == 2, "observation reports per-prediction set counts") &&
        check(result->lead_us >= 0, "lead time uses nonnegative steady-clock duration") &&
        check(!state.observe(1, 1, 7, ids({ 4, 8 })).has_value(), "prediction did not cross sequences") &&
        check(state.stats().positions == 3, "batch positions counted once per row");
}

bool test_staging_row_selection() {
    for (size_t rows = 0; rows <= llama_prerouter_state::max_positions; ++rows) {
        size_t staging_predictions = 0;
        size_t shadow_predictions = 0;
        for (size_t row = 0; row <= rows; ++row) {
            staging_predictions += llama_prerouter_state::should_predict_row(true, rows, row);
            shadow_predictions += llama_prerouter_state::should_predict_row(false, rows, row);
        }
        if (!check(staging_predictions == (rows ? 1 : 0), "one staging prediction per nonempty graph")) return false;
        if (!check(shadow_predictions == rows, "shadow retains every verification row")) return false;
        if (rows > 1 && !check(llama_prerouter_state::should_predict_row(true, rows, rows - 2),
                    "MTP staging predicts the final in-batch consumer")) return false;
    }
    llama_prerouter_state state;
    if (!check(state.begin_batch({ { 0, 10 }, { 0, 11 }, { 0, 12 }, { 0, 13 } }), "MTP rows accepted")) return false;
    if (!check(state.predict(0, 12, 6, ids({ 1, 3, 5, 7 })), "penultimate row stages next-position route")) return false;
    if (!check(!state.observe(0, 12, 7, ids({ 1, 3, 5, 7 })), "prediction is not used for its own position")) return false;
    const auto result = state.observe(0, 13, 7, ids({ 7, 5, 3, 1 }));
    return check(result && result->matched, "selected hint matches final row of the same graph");
}

bool test_expert_set_recall() {
    llama_prerouter_state state;
    if (!check(state.begin_batch({ { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 } }),
                "expert recall batch accepted")) return false;

    if (!check(state.predict(0, 0, 3, ids({ 1, 2, 2, 4 })), "overlap prediction staged")) return false;
    const auto overlap = state.observe(0, 1, 4, ids({ 2, 4, 8, 8 }));
    if (!check(overlap && !overlap->matched && overlap->hits == 2 && overlap->total == 3,
                "overlap reports unique expert recall")) return false;

    if (!check(state.predict(0, 1, 3, ids({ 1, 2, 3, 4, 4 })), "unordered prediction staged")) return false;
    const auto reversed = state.observe(0, 2, 4, ids({ 4, 3, 2, 1, 1 }));
    if (!check(reversed && reversed->matched && reversed->hits == 4 && reversed->total == 4,
                "order and duplicates do not affect exact-set match")) return false;

    if (!check(state.predict(0, 2, 3, ids({ 2, 2, 2, 3 })), "duplicate prediction staged")) return false;
    const auto duplicate = state.observe(0, 3, 4, ids({ 2, 2, 3, 3, 3 }));
    if (!check(duplicate && duplicate->matched && duplicate->hits == 2 && duplicate->total == 2,
                "duplicates cannot inflate hit or total counts")) return false;

    const auto stats = state.stats();
    return check(stats.comparisons == 3 && stats.hits == 8 && stats.expert_positions == 9,
            "stats accumulate comparison, intersection, and predicted-expert counts");
}

bool test_rollback_and_replay() {
    llama_prerouter_state state;
    if (!check(state.begin_batch({ { 2, 0 }, { 2, 1 }, { 2, 2 }, { 2, 3 } }), "initial sequence batch accepted")) return false;
    if (!check(state.route(2, 1, 5, ids({ 11 })), "pre-rollback route stored")) return false;
    if (!check(state.route(2, 2, 5, ids({ 12 })), "future route stored")) return false;
    if (!check(state.predict(2, 1, 5, ids({ 21 })), "prediction before rollback boundary staged")) return false;
    if (!check(state.predict(2, 2, 5, ids({ 22 })), "prediction at rollback boundary staged")) return false;

    if (!check(state.begin_batch({ { 2, 2 } }), "rollback batch accepted")) return false;
    if (!check(state.previous_route(2, 2, 5) == ids({ 11 }), "rollback retains history before first replayed position")) return false;
    if (!check(state.route(2, 2, 5, ids({ 12 })), "replayed route stored")) return false;
    const auto retained = state.observe(2, 2, 6, ids({ 21 }));
    if (!check(retained && retained->matched, "prediction originating before rollback boundary retained")) return false;
    if (!check(!state.observe(2, 3, 6, ids({ 22 })).has_value(), "prediction from rejected future dropped")) return false;
    return check(state.stats().invalidations == 1, "rollback invalidation counted");
}

bool test_continuity_and_gaps() {
    llama_prerouter_state state;
    if (!check(state.begin_batch({ { 4, 10 } }), "initial continuity batch accepted")) return false;
    if (!check(state.route(4, 10, 3, ids({ 7 })), "continuity route stored")) return false;
    if (!check(state.predict(4, 10, 3, ids({ 8 })), "continuity prediction staged")) return false;
    if (!check(state.begin_batch({ { 4, 11 } }), "contiguous batch accepted")) return false;
    if (!check(state.previous_route(4, 11, 3) == ids({ 7 }), "contiguous batch retains history")) return false;

    if (!check(state.begin_batch({ { 4, 13 } }), "gap batch accepted")) return false;
    if (!check(state.previous_route(4, 13, 3).empty(), "gap clears route history")) return false;
    if (!check(!state.observe(4, 11, 4, ids({ 8 })).has_value(), "gap clears staged predictions")) return false;
    return check(state.stats().invalidations == 1, "gap invalidation counted");
}

bool test_sequence_and_layer_isolation() {
    llama_prerouter_state state;
    if (!check(state.begin_batch({ { 0, 0 }, { 1, 0 }, { 0, 1 }, { 0, 2 } }), "two sequence batch accepted")) return false;
    if (!check(state.predict(0, 0, 8, ids({ 1, 2 })), "seq 0 prediction staged")) return false;
    if (!check(!state.observe(1, 1, 9, ids({ 1, 2 })).has_value(), "prediction did not cross sequences")) return false;
    if (!check(!state.observe(0, 1, 8, ids({ 1, 2 })).has_value(), "layer mismatch not compared")) return false;
    const auto match = state.observe(0, 1, 9, ids({ 1, 2 }));
    if (!check(match && match->matched, "exact sequence and layer match consumed")) return false;
    if (!check(state.predict(0, 1, 8, ids({ 1, 2 })), "rejected-route prediction staged")) return false;
    const auto result = state.observe(0, 2, 9, ids({ 0 }));
    if (!check(result && !result->matched, "rejected draft route is counted as a comparison")) return false;
    return check(result->total == 2 && result->hits == 0, "rejected draft does not claim a hit") &&
        check(match->total == 2 && match->hits == 2, "per-observation set counts reported") &&
        check(state.stats().comparisons == 2 && state.stats().hits == 2 &&
                state.stats().expert_positions == 4, "aggregate recall statistics accumulated");
}

bool test_bounds_validation_and_reset() {
    llama_prerouter_state state;
    std::vector<std::pair<int32_t, int32_t>> sequences;
    for (int32_t seq = 0; seq < llama_prerouter_state::max_sequences; ++seq) {
        sequences.push_back({ seq, 0 });
    }
    if (!check(state.begin_batch(sequences), "32 sequences accepted")) return false;
    if (!check(!state.begin_batch({ { 32, 0 } }), "33rd sequence rejected")) return false;

    std::vector<std::pair<int32_t, int32_t>> positions;
    for (int32_t pos = 0; pos < 10; ++pos) {
        positions.push_back({ 0, pos });
    }
    if (!check(state.begin_batch(positions), "bounded route history batch accepted")) return false;
    for (int32_t pos = 0; pos < 10; ++pos) {
        if (!check(state.route(0, pos, 0, ids({ pos })), "bounded route stored")) return false;
    }
    if (!check(state.previous_route(0, 9, 0) == ids({ 8 }), "last eight route positions retained")) return false;
    if (!check(state.previous_route(0, 2, 0).empty(), "older route position evicted")) return false;

    for (int32_t origin = 0; origin < 9; ++origin) {
        if (!check(state.predict(0, origin, 0, ids({ origin })), "bounded prediction staged")) return false;
    }
    if (!check(!state.observe(0, 1, 1, ids({ 0 })).has_value(), "oldest prediction position evicted")) return false;
    if (!check(state.observe(0, 2, 1, ids({ 1 })).has_value(), "eight prediction positions retained")) return false;

    state.reset();
    if (!check(state.stats().comparisons == 0 && state.stats().hits == 0 &&
                state.stats().positions == 0 && state.stats().expert_positions == 0 &&
                state.stats().invalidations == 0,
                "reset clears counters")) return false;
    if (!check(state.begin_batch({ { 32, 0 } }), "reset releases sequence capacity")) return false;
    return check(state.previous_route(32, 1, 0).empty(), "reset clears route history");
}

bool test_invalid_batches_are_atomic() {
    llama_prerouter_state state;
    if (!check(state.begin_batch({ { 5, 0 } }), "valid setup batch accepted")) return false;
    if (!check(state.route(5, 0, 2, ids({ 6 })), "setup route stored")) return false;
    if (!check(!state.begin_batch({ { 5, 1 }, { 5, 1 } }), "duplicate per-sequence positions rejected")) return false;
    if (!check(!state.begin_batch({ { 5, 1 }, { -1, 0 } }), "invalid sequence id rejected")) return false;
    return check(state.previous_route(5, 1, 2) == ids({ 6 }), "rejected batches leave state unchanged");
}

} // namespace

int main() {
    return test_batched_alignment_and_interleaving() &&
            test_staging_row_selection() &&
            test_expert_set_recall() &&
            test_rollback_and_replay() &&
            test_continuity_and_gaps() &&
            test_sequence_and_layer_isolation() &&
            test_bounds_validation_and_reset() &&
            test_invalid_batches_are_atomic() ? 0 : 1;
}
