#include "llama-prerouter.h"

// Checkpoint layout and head architecture: Edge0 (Apache-2.0),
// https://github.com/Edge0-AI/edge0/blob/main/docs/prerouter.md.

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

using json = nlohmann::json;

constexpr int k_model_layers = 40;
constexpr int k_hidden = 2048;
constexpr int k_experts = 256;
constexpr int k_top_k = 4;
constexpr int k_head_hidden = 512;
constexpr int k_features = k_hidden + 2 * k_experts;
constexpr uint64_t k_max_header_size = 64ULL * 1024 * 1024;

struct tensor_info {
    uint64_t begin;
    uint64_t end;
};

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool read_u64_le(std::ifstream & file, uint64_t & value) {
    unsigned char bytes[8];
    if (!file.read(reinterpret_cast<char *>(bytes), sizeof(bytes))) {
        return false;
    }
    value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= uint64_t(bytes[i]) << (i * 8);
    }
    return true;
}

bool get_tensor(const json & header, const std::string & name,
                const std::vector<uint64_t> & expected_shape,
                uint64_t data_size, tensor_info & result, std::string & error) {
    const auto it = header.find(name);
    if (it == header.end() || !it->is_object()) {
        return fail(error, "missing tensor " + name);
    }
    const auto dtype = it->find("dtype");
    const auto shape = it->find("shape");
    const auto offsets = it->find("data_offsets");
    if (dtype == it->end() || !dtype->is_string() || dtype->get<std::string>() != "F16") {
        return fail(error, "tensor " + name + " must have dtype F16");
    }
    if (shape == it->end() || !shape->is_array() || shape->size() != expected_shape.size()) {
        return fail(error, "tensor " + name + " has an invalid shape");
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (!(*shape)[i].is_number_unsigned() && !(*shape)[i].is_number_integer()) {
            return fail(error, "tensor " + name + " has an invalid shape");
        }
        if ((*shape)[i].get<int64_t>() != static_cast<int64_t>(expected_shape[i])) {
            return fail(error, "tensor " + name + " has an unexpected shape");
        }
    }
    if (offsets == it->end() || !offsets->is_array() || offsets->size() != 2 ||
        !(*offsets)[0].is_number_unsigned() || !(*offsets)[1].is_number_unsigned()) {
        return fail(error, "tensor " + name + " has invalid data offsets");
    }
    result.begin = (*offsets)[0].get<uint64_t>();
    result.end = (*offsets)[1].get<uint64_t>();

    uint64_t elements = 1;
    for (uint64_t dim : expected_shape) {
        if (dim != 0 && elements > std::numeric_limits<uint64_t>::max() / dim) {
            return fail(error, "tensor " + name + " shape overflows");
        }
        elements *= dim;
    }
    if (elements > std::numeric_limits<uint64_t>::max() / sizeof(uint16_t) ||
        result.begin > result.end || result.end > data_size ||
        result.end - result.begin != elements * sizeof(uint16_t)) {
        return fail(error, "tensor " + name + " data range is invalid");
    }
    return true;
}

float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x03ff;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ff;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 15 - shift + 1);
            bits = sign | (exp32 << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool valid_expert_ids(const std::vector<int32_t> & ids, int count, int n_experts) {
    if (ids.size() != static_cast<size_t>(count)) {
        return false;
    }
    for (int32_t id : ids) {
        if (id < 0 || id >= n_experts) {
            return false;
        }
    }
    return true;
}

float dot_product(const float * weights, const float * features, size_t count) {
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    for (size_t i = 0; i < count; i += 4) {
        s0 += weights[i]     * features[i];
        s1 += weights[i + 1] * features[i + 1];
        s2 += weights[i + 2] * features[i + 2];
        s3 += weights[i + 3] * features[i + 3];
    }
    return (s0 + s1) + (s2 + s3);
}

} // namespace

llama_prerouter::llama_prerouter(int n_layers, int n_embd, int n_experts, int n_expert_used)
    : n_layers_(n_layers), n_embd_(n_embd), n_experts_(n_experts),
      n_expert_used_(n_expert_used), owners_(n_layers) {
}

std::unique_ptr<llama_prerouter> llama_prerouter::load(
        const std::string & path, int n_layers, int n_embd, int n_experts,
        int n_expert_used, std::string & error) {
    error.clear();
    if (n_layers != k_model_layers || n_embd != k_hidden || n_experts != k_experts ||
        n_expert_used != k_top_k) {
        error = "prerouter only supports Qwen3.6-35B-A3B (40 layers, 2048 hidden, 256 experts, K=4)";
        return nullptr;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open prerouter checkpoint: " + path;
        return nullptr;
    }
    uint64_t header_size = 0;
    if (!read_u64_le(file, header_size) || header_size == 0 || header_size > k_max_header_size) {
        error = "invalid safetensors header length";
        return nullptr;
    }
    std::string header_text(static_cast<size_t>(header_size), '\0');
    if (!file.read(header_text.data(), static_cast<std::streamsize>(header_text.size()))) {
        error = "truncated safetensors header";
        return nullptr;
    }

    json header;
    try {
        header = json::parse(header_text);
    } catch (const std::exception & e) {
        error = std::string("invalid safetensors JSON header: ") + e.what();
        return nullptr;
    }
    if (!header.is_object()) {
        error = "safetensors header must be an object";
        return nullptr;
    }
    const auto metadata_it = header.find("__metadata__");
    if (metadata_it == header.end() || !metadata_it->is_object()) {
        error = "checkpoint is missing __metadata__";
        return nullptr;
    }
    const auto nested_it = metadata_it->find("__metadata__");
    if (nested_it == metadata_it->end() || !nested_it->is_string()) {
        error = "checkpoint is missing serialized Edge0 metadata";
        return nullptr;
    }
    json metadata;
    try {
        metadata = json::parse(nested_it->get<std::string>());
    } catch (const std::exception & e) {
        error = std::string("invalid Edge0 metadata JSON: ") + e.what();
        return nullptr;
    }
    if (!metadata.is_object() || !metadata.contains("model") || !metadata["model"].is_string() ||
        !metadata.contains("kind") || !metadata["kind"].is_string() ||
        !metadata.contains("K") || !metadata["K"].is_string() ||
        metadata["model"].get<std::string>() != "edge0-35b" ||
        metadata["kind"].get<std::string>() != "prerouter" ||
        metadata["K"].get<std::string>() != "4") {
        error = "checkpoint metadata must identify edge0-35b/prerouter/K=4";
        return nullptr;
    }
    if (!metadata.contains("owners") || !metadata["owners"].is_string()) {
        error = "checkpoint metadata is missing the owner list";
        return nullptr;
    }
    try {
        const json owners = json::parse(metadata["owners"].get<std::string>());
        if (!owners.is_array() || owners.size() != 33) {
            error = "checkpoint metadata must declare all 33 owners (6..38)";
            return nullptr;
        }
        for (int i = 0; i < 33; ++i) {
            if (!owners[i].is_number_integer() || owners[i].get<int>() != i + 6) {
                error = "checkpoint metadata must declare all 33 owners (6..38)";
                return nullptr;
            }
        }
    } catch (const std::exception & e) {
        error = std::string("invalid owner list in Edge0 metadata: ") + e.what();
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff file_end = file.tellg();
    if (file_end < 0 || static_cast<uint64_t>(file_end) < 8 + header_size) {
        error = "checkpoint file is shorter than its safetensors header";
        return nullptr;
    }
    const uint64_t data_size = static_cast<uint64_t>(file_end) - 8 - header_size;

    auto model = std::unique_ptr<llama_prerouter>(
        new llama_prerouter(n_layers, n_embd, n_experts, n_expert_used));
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    const uint64_t fc1_count = uint64_t(k_head_hidden) * k_features;
    const uint64_t fc2_count = uint64_t(k_experts) * k_head_hidden;
    const uint64_t linear_count = uint64_t(k_experts) * k_features;
    for (int owner = 6; owner <= 38; ++owner) {
        auto & weights = model->owners_[owner];
        const std::string prefix = "layers." + std::to_string(owner) + ".";
        tensor_info t1{}, t2{}, tl{};
        if (!get_tensor(header, prefix + "fc1.weight", {k_head_hidden, k_features}, data_size, t1, error) ||
            !get_tensor(header, prefix + "fc2.weight", {k_experts, k_head_hidden}, data_size, t2, error) ||
            !get_tensor(header, prefix + "linear_init.weight", {k_experts, k_features}, data_size, tl, error)) {
            return nullptr;
        }
        ranges.emplace_back(t1.begin, t1.end);
        ranges.emplace_back(t2.begin, t2.end);
        ranges.emplace_back(tl.begin, tl.end);
        auto read_weights = [&](const tensor_info & tensor, uint64_t count,
                                std::vector<float> & destination, const std::string & name) {
            std::vector<uint16_t> raw(static_cast<size_t>(count));
            destination.resize(static_cast<size_t>(count));
            file.clear();
            file.seekg(static_cast<std::streamoff>(8 + header_size + tensor.begin), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(raw.data()),
                           static_cast<std::streamsize>(count * sizeof(uint16_t)))) {
                error = "truncated tensor data for " + name;
                return false;
            }
            for (size_t i = 0; i < raw.size(); ++i) {
                destination[i] = half_to_float(raw[i]);
            }
            return true;
        };
        if (!read_weights(t1, fc1_count, weights.fc1, prefix + "fc1.weight") ||
            !read_weights(t2, fc2_count, weights.fc2, prefix + "fc2.weight") ||
            !read_weights(tl, linear_count, weights.linear_init, prefix + "linear_init.weight")) {
            return nullptr;
        }
    }

    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
            error = "checkpoint tensor data ranges overlap";
            return nullptr;
        }
    }
    return model;
}

bool llama_prerouter::has_owner(int layer) const {
    return layer >= 0 && layer < n_layers_ && !owners_[layer].fc1.empty();
}

std::vector<int32_t> llama_prerouter::predict(
        int owner_layer, const float * hidden, size_t hidden_count,
        const std::vector<int32_t> & current_experts,
        const std::vector<int32_t> & previous_experts) const {
    if (!has_owner(owner_layer) || hidden == nullptr || hidden_count != static_cast<size_t>(n_embd_) ||
        !valid_expert_ids(current_experts, n_expert_used_, n_experts_) ||
        (!previous_experts.empty() && !valid_expert_ids(previous_experts, n_expert_used_, n_experts_))) {
        return {};
    }

    std::vector<float> features(k_features, 0.0f);
    for (int i = 0; i < n_embd_; ++i) {
        if (!std::isfinite(hidden[i])) {
            return {};
        }
        features[i] = hidden[i];
    }
    for (int32_t id : current_experts) {
        features[n_embd_ + id] = 1.0f;
    }
    for (int32_t id : previous_experts) {
        features[n_embd_ + n_experts_ + id] = 1.0f;
    }

    const auto & weights = owners_[owner_layer];
    std::vector<float> activation(k_head_hidden);
    for (int row = 0; row < k_head_hidden; ++row) {
        const size_t offset = static_cast<size_t>(row) * k_features;
        const float value = dot_product(weights.fc1.data() + offset, features.data(), k_features);
        if (!std::isfinite(value)) {
            return {};
        }
        activation[row] = 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
        if (!std::isfinite(activation[row])) {
            return {};
        }
    }

    std::vector<float> logits(n_experts_);
    for (int row = 0; row < n_experts_; ++row) {
        const size_t linear_offset = static_cast<size_t>(row) * k_features;
        const size_t fc2_offset = static_cast<size_t>(row) * k_head_hidden;
        const float residual = dot_product(weights.linear_init.data() + linear_offset, features.data(), k_features);
        const float projected = dot_product(weights.fc2.data() + fc2_offset, activation.data(), k_head_hidden);
        logits[row] = residual + projected;
        if (!std::isfinite(logits[row])) {
            return {};
        }
    }

    std::vector<int32_t> selected(n_experts_);
    for (int i = 0; i < n_experts_; ++i) {
        selected[i] = i;
    }
    std::partial_sort(selected.begin(), selected.begin() + n_expert_used_, selected.end(),
        [&](int32_t a, int32_t b) {
            return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
        });
    selected.resize(n_expert_used_);
    return selected;
}
