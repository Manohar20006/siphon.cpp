#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

// Small, model-independent state for checking prerouter shadow predictions.
class llama_prerouter_state {
public:
    static constexpr int32_t max_sequences = 32;
    static constexpr int32_t max_layers = 40;
    static constexpr size_t max_positions = 8;

    // In a verification batch, the penultimate row predicts a route that will
    // actually be evaluated later in this graph. The last speculative row's
    // successor may be discarded by MTP rollback. Shadow mode still evaluates
    // every row for diagnostics; staging mode pays for just one useful hint.
    static bool should_predict_row(bool prefetch, size_t rows, size_t row) {
        return row < rows && (!prefetch || row == (rows > 1 ? rows - 2 : 0));
    }

    struct stats_type {
        uint64_t comparisons = 0;      // staged predictions consumed by observe
        uint64_t hits = 0;             // unique predicted experts found in actual routes
        uint64_t positions = 0;        // accepted rows passed to begin_batch
        uint64_t expert_positions = 0; // unique predicted experts compared
        uint64_t invalidations = 0;    // staged predictions discarded before observation
    };

    struct observation_result {
        bool matched = false;
        uint64_t hits = 0;  // unique predicted experts also present in actual
        uint64_t total = 0; // unique predicted experts
        int64_t lead_us = 0;
    };

    // Rows may interleave sequences, but positions for each sequence must be
    // strictly increasing. A rejected batch leaves this state unchanged.
    bool begin_batch(const std::vector<std::pair<int32_t, int32_t>> & rows) {
        struct batch_sequence {
            int32_t seq;
            int32_t first_pos;
            int32_t last_pos;
            bool discontinuity;
        };

        std::vector<batch_sequence> batch_sequences;
        for (const auto & row : rows) {
            const int32_t seq = row.first;
            const int32_t pos = row.second;
            if (seq < 0 || pos < 0) {
                return false;
            }
            auto it = std::find_if(batch_sequences.begin(), batch_sequences.end(),
                    [seq](const batch_sequence & entry) { return entry.seq == seq; });
            if (it == batch_sequences.end()) {
                if (batch_sequences.size() == static_cast<size_t>(max_sequences)) {
                    return false;
                }
                batch_sequences.push_back({ seq, pos, pos, false });
            } else {
                if (pos <= it->last_pos) {
                    return false;
                }
                if (int64_t(pos) != int64_t(it->last_pos) + 1) {
                    it->discontinuity = true;
                }
                it->last_pos = pos;
            }
        }

        size_t new_sequences = 0;
        for (const auto & entry : batch_sequences) {
            if (find_sequence(entry.seq) == nullptr) {
                ++new_sequences;
            }
        }
        if (sequences_.size() + new_sequences > static_cast<size_t>(max_sequences)) {
            return false;
        }

        for (const auto & entry : batch_sequences) {
            sequence_state * state = find_sequence(entry.seq);
            if (state == nullptr) {
                sequences_.push_back(sequence_state{ entry.seq, 0, false, {}, {} });
                state = &sequences_.back();
            } else if (state->has_frontier) {
                if (entry.first_pos <= state->frontier) {
                    trim_from(*state, entry.first_pos);
                } else if (int64_t(entry.first_pos) != int64_t(state->frontier) + 1) {
                    clear_history(*state);
                }
            }

            if (entry.discontinuity) {
                clear_history(*state);
            }
            state->frontier = entry.last_pos;
            state->has_frontier = true;
        }
        stats_.positions += rows.size();
        return true;
    }

    // Store the actual route for one layer and sequence position.
    bool route(int32_t seq, int32_t pos, int32_t layer,
            const std::vector<int32_t> & ids) {
        if (pos < 0 || !valid_layer(layer)) {
            return false;
        }
        sequence_state * state = find_sequence(seq);
        if (state == nullptr) {
            return false;
        }

        auto position = std::find_if(state->routes.begin(), state->routes.end(),
                [pos](const route_position & entry) { return entry.pos == pos; });
        if (position == state->routes.end()) {
            position = state->routes.insert(
                    std::lower_bound(state->routes.begin(), state->routes.end(), pos,
                            [](const route_position & entry, int32_t value) {
                                return entry.pos < value;
                            }),
                    route_position{ pos, {} });
        }

        auto layer_route = std::find_if(position->layers.begin(), position->layers.end(),
                [layer](const route_entry & entry) { return entry.layer == layer; });
        if (layer_route == position->layers.end()) {
            position->layers.push_back({ layer, ids });
        } else {
            layer_route->ids = ids;
        }

        while (state->routes.size() > max_positions) {
            state->routes.erase(state->routes.begin());
        }
        return true;
    }

    // Return the same layer's route from the immediately preceding position.
    std::vector<int32_t> previous_route(int32_t seq, int32_t pos,
            int32_t owner) const {
        if (pos <= 0 || !valid_layer(owner)) {
            return {};
        }
        const sequence_state * state = find_sequence(seq);
        if (state == nullptr) {
            return {};
        }
        const int32_t previous_pos = pos - 1;
        const auto position = std::find_if(state->routes.begin(), state->routes.end(),
                [previous_pos](const route_position & entry) {
                    return entry.pos == previous_pos;
                });
        if (position == state->routes.end()) {
            return {};
        }
        const auto layer_route = std::find_if(position->layers.begin(), position->layers.end(),
                [owner](const route_entry & entry) { return entry.layer == owner; });
        return layer_route == position->layers.end() ? std::vector<int32_t>{} : layer_route->ids;
    }

    // Stage owner+1's prediction for the next position.
    bool predict(int32_t seq, int32_t origin_pos, int32_t owner,
            const std::vector<int32_t> & ids) {
        const int64_t target_layer = int64_t(owner) + 1;
        if (origin_pos < 0 || owner < 0 || target_layer >= max_layers ||
                origin_pos == INT32_MAX) {
            return false;
        }
        sequence_state * state = find_sequence(seq);
        if (state == nullptr) {
            return false;
        }

        auto prediction = std::find_if(state->predictions.begin(), state->predictions.end(),
                [origin_pos, target_layer](const staged_prediction & entry) {
                    return entry.origin_pos == origin_pos && entry.target_layer == target_layer;
                });
        if (prediction == state->predictions.end()) {
            state->predictions.push_back({ origin_pos, static_cast<int32_t>(target_layer), ids,
                    clock_type::now() });
        } else {
            prediction->ids = ids;
            prediction->staged_at = clock_type::now();
        }
        trim_predictions(*state);
        return true;
    }

    // A result is present only when this exact sequence/position/layer had a
    // staged prediction. Expert IDs are compared as sets, so ordering and
    // duplicate IDs do not affect recall or an exact-set match.
    std::optional<observation_result> observe(int32_t seq, int32_t pos,
            int32_t target_layer, const std::vector<int32_t> & actual) {
        if (pos <= 0 || !valid_layer(target_layer)) {
            return std::nullopt;
        }
        sequence_state * state = find_sequence(seq);
        if (state == nullptr) {
            return std::nullopt;
        }
        const int32_t origin_pos = pos - 1;
        auto prediction = std::find_if(state->predictions.begin(), state->predictions.end(),
                [origin_pos, target_layer](const staged_prediction & entry) {
                    return entry.origin_pos == origin_pos && entry.target_layer == target_layer;
                });
        if (prediction == state->predictions.end()) {
            return std::nullopt;
        }

        const clock_type::time_point staged_at = prediction->staged_at;
        std::vector<int32_t> predicted_ids = prediction->ids;
        std::vector<int32_t> actual_ids = actual;
        std::sort(predicted_ids.begin(), predicted_ids.end());
        predicted_ids.erase(std::unique(predicted_ids.begin(), predicted_ids.end()), predicted_ids.end());
        std::sort(actual_ids.begin(), actual_ids.end());
        actual_ids.erase(std::unique(actual_ids.begin(), actual_ids.end()), actual_ids.end());
        std::vector<int32_t> intersection;
        intersection.reserve(std::min(predicted_ids.size(), actual_ids.size()));
        std::set_intersection(predicted_ids.begin(), predicted_ids.end(),
                actual_ids.begin(), actual_ids.end(), std::back_inserter(intersection));
        const uint64_t hits = intersection.size();
        const uint64_t total = predicted_ids.size();
        const bool matched = hits == total && actual_ids.size() == predicted_ids.size();
        state->predictions.erase(prediction);
        ++stats_.comparisons;
        stats_.hits += hits;
        stats_.expert_positions += total;
        const int64_t lead_us = std::chrono::duration_cast<std::chrono::microseconds>(
                clock_type::now() - staged_at).count();
        return observation_result{ matched, hits, total, lead_us };
    }

    stats_type stats() const {
        return stats_;
    }

    void reset() {
        sequences_.clear();
        stats_ = {};
    }

private:
    using clock_type = std::chrono::steady_clock;

    struct route_entry {
        int32_t layer;
        std::vector<int32_t> ids;
    };

    struct route_position {
        int32_t pos;
        std::vector<route_entry> layers;
    };

    struct staged_prediction {
        int32_t origin_pos;
        int32_t target_layer;
        std::vector<int32_t> ids;
        clock_type::time_point staged_at;
    };

    struct sequence_state {
        int32_t seq;
        int32_t frontier = 0;
        bool has_frontier = false;
        std::vector<route_position> routes;
        std::vector<staged_prediction> predictions;
    };

    static bool valid_layer(int32_t layer) {
        return layer >= 0 && layer < max_layers;
    }

    sequence_state * find_sequence(int32_t seq) {
        const auto it = std::find_if(sequences_.begin(), sequences_.end(),
                [seq](const sequence_state & entry) { return entry.seq == seq; });
        return it == sequences_.end() ? nullptr : &*it;
    }

    const sequence_state * find_sequence(int32_t seq) const {
        const auto it = std::find_if(sequences_.begin(), sequences_.end(),
                [seq](const sequence_state & entry) { return entry.seq == seq; });
        return it == sequences_.end() ? nullptr : &*it;
    }

    void trim_from(sequence_state & state, int32_t first_pos) {
        state.routes.erase(std::remove_if(state.routes.begin(), state.routes.end(),
                [first_pos](const route_position & entry) { return entry.pos >= first_pos; }),
                state.routes.end());
        const size_t before = state.predictions.size();
        state.predictions.erase(std::remove_if(state.predictions.begin(), state.predictions.end(),
                [first_pos](const staged_prediction & entry) {
                    return entry.origin_pos >= first_pos;
                }), state.predictions.end());
        stats_.invalidations += before - state.predictions.size();
    }

    void clear_history(sequence_state & state) {
        state.routes.clear();
        stats_.invalidations += state.predictions.size();
        state.predictions.clear();
    }

    void trim_predictions(sequence_state & state) {
        std::vector<int32_t> origins;
        origins.reserve(state.predictions.size());
        for (const auto & prediction : state.predictions) {
            origins.push_back(prediction.origin_pos);
        }
        std::sort(origins.begin(), origins.end());
        origins.erase(std::unique(origins.begin(), origins.end()), origins.end());
        if (origins.size() <= max_positions) {
            return;
        }
        const int32_t first_kept = origins[origins.size() - max_positions];
        const size_t before = state.predictions.size();
        state.predictions.erase(std::remove_if(state.predictions.begin(), state.predictions.end(),
                [first_kept](const staged_prediction & entry) {
                    return entry.origin_pos < first_kept;
                }), state.predictions.end());
        stats_.invalidations += before - state.predictions.size();
    }

    std::vector<sequence_state> sequences_;
    stats_type stats_;
};
