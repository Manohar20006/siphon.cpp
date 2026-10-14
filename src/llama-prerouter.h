#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class llama_prerouter {
public:
    static std::unique_ptr<llama_prerouter> load(
            const std::string & path,
            int n_layers,
            int n_embd,
            int n_experts,
            int n_expert_used,
            std::string & error);

    static std::shared_ptr<const llama_prerouter> load_shared(
            const std::string & path,
            int n_layers,
            int n_embd,
            int n_experts,
            int n_expert_used,
            std::string & error);

    bool has_owner(int layer) const;

    std::vector<int32_t> predict(
            int owner_layer,
            const float * hidden,
            size_t hidden_count,
            const std::vector<int32_t> & current_experts,
            const std::vector<int32_t> & previous_experts) const;

private:
    struct owner_weights {
        std::vector<float> fc1;
        std::vector<float> fc2;
        std::vector<float> linear_init;
    };

    llama_prerouter(int n_layers, int n_embd, int n_experts, int n_expert_used);

    int n_layers_;
    int n_embd_;
    int n_experts_;
    int n_expert_used_;
    std::vector<owner_weights> owners_;
};
