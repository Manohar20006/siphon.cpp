#include "ggml.h"
#include "gguf.h"
#include "llama-dstorage-slots.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

static std::wstring widen_path(const std::string & path) {
    return std::wstring(path.begin(), path.end());
}

static bool register_tensor(
        DStorageSlotManager & slots,
        gguf_context * ctx,
        const std::string & model_path,
        int layer,
        const char * name,
        int n_experts,
        uint64_t * out_size) {
    const int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) {
        fprintf(stderr, "missing tensor: %s\n", name);
        return false;
    }

    const uint64_t abs_offset = uint64_t(gguf_get_data_offset(ctx)) + uint64_t(gguf_get_tensor_offset(ctx, tid));
    const uint64_t size       = uint64_t(gguf_get_tensor_size(ctx, tid));
    if (out_size != nullptr) {
        *out_size = size;
    }
    slots.register_expert_tensor(layer, name, widen_path(model_path), abs_offset, size, n_experts);
    fprintf(stderr, "registered %s abs_offset=%llu size=%.2f MiB stride=%.4f MiB\n",
            name,
            (unsigned long long) abs_offset,
            size / 1024.0 / 1024.0,
            (size / (double) n_experts) / 1024.0 / 1024.0);
    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [layer] [n_experts_to_stream] [warm_pool_first] [pinned_cache_mib] [repeat_load] [gpu_cache_mib]\n", argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];
    const int layer = argc >= 3 ? std::atoi(argv[2]) : 0;
    const int n_stream_arg = argc >= 4 ? std::atoi(argv[3]) : 8;
    const bool warm_pool_first = argc >= 5 ? std::atoi(argv[4]) != 0 : false;
    const int pinned_cache_mib = argc >= 6 ? std::atoi(argv[5]) : 0;
    const bool repeat_load = argc >= 7 ? std::atoi(argv[6]) != 0 : false;
    const int gpu_cache_mib = argc >= 8 ? std::atoi(argv[7]) : 1024;

    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to open GGUF: %s\n", model_path.c_str());
        return 1;
    }

    // Detect Gemma vs Qwen
    char check_name[128];
    snprintf(check_name, sizeof(check_name), "blk.%d.ffn_gate_up_exps.weight", layer);
    const bool is_gemma = gguf_find_tensor(ctx, check_name) >= 0;
    const int n_experts = is_gemma ? 128 : 256;
    
    fprintf(stderr, "detected model architecture: %s, setting n_experts=%d\n", is_gemma ? "Gemma" : "Qwen", n_experts);

    int n_stream = n_stream_arg;
    if (n_stream <= 0 || n_stream > n_experts) {
        fprintf(stderr, "n_experts_to_stream must be in [1, %d], got %d\n", n_experts, n_stream);
        gguf_free(ctx);
        return 2;
    }

    DStorageSlotManager slots;
    if (!slots.init(30, n_experts, 8, uint32_t(gpu_cache_mib), uint32_t(pinned_cache_mib))) {
        fprintf(stderr, "slot manager init failed\n");
        gguf_free(ctx);
        return 1;
    }

    uint64_t total_size = 0;
    if (is_gemma) {
        char down_name[128];
        char gate_up_name[128];
        snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", layer);
        snprintf(gate_up_name, sizeof(gate_up_name), "blk.%d.ffn_gate_up_exps.weight", layer);

        uint64_t down_size = 0;
        uint64_t gate_up_size = 0;
        if (!register_tensor(slots, ctx, model_path, layer, down_name, n_experts, &down_size) ||
            !register_tensor(slots, ctx, model_path, layer, gate_up_name, n_experts, &gate_up_size)) {
            slots.destroy();
            gguf_free(ctx);
            return 1;
        }
        total_size = down_size + gate_up_size;
    } else {
        char down_name[128];
        char gate_name[128];
        char up_name[128];
        snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", layer);
        snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gate_exps.weight", layer);
        snprintf(up_name, sizeof(up_name), "blk.%d.ffn_up_exps.weight", layer);

        uint64_t down_size = 0;
        uint64_t gate_size = 0;
        uint64_t up_size = 0;
        if (!register_tensor(slots, ctx, model_path, layer, down_name, n_experts, &down_size) ||
            !register_tensor(slots, ctx, model_path, layer, gate_name, n_experts, &gate_size) ||
            !register_tensor(slots, ctx, model_path, layer, up_name, n_experts, &up_size)) {
            slots.destroy();
            gguf_free(ctx);
            return 1;
        }
        total_size = down_size + gate_size + up_size;
    }

    std::vector<int32_t> expert_ids(n_stream);
    for (int i = 0; i < n_stream; ++i) {
        expert_ids[i] = i;
    }

    std::unordered_map<std::string, uint64_t> pool_ptrs;
    std::vector<int32_t> id_map;

    if (warm_pool_first) {
        std::vector<int32_t> warm_ids = {0}; // use expert 0 for warm ID
        std::unordered_map<std::string, uint64_t> warm_pool_ptrs;
        std::vector<int32_t> warm_id_map;
        if (!slots.ensure_experts_loaded(layer, warm_ids.data(), int(warm_ids.size()), warm_pool_ptrs, warm_id_map)) {
            fprintf(stderr, "warm pool allocation failed\n");
            slots.destroy();
            gguf_free(ctx);
            return 1;
        }
    }

    const int64_t t0 = ggml_time_us();
    const bool load_ok = slots.ensure_experts_loaded(layer, expert_ids.data(), int(expert_ids.size()), pool_ptrs, id_map);
    const bool direct_execution = slots.get_active_slot_count() == 0;
    const bool ok = load_ok && direct_execution;
    const int64_t elapsed_us = ggml_time_us() - t0;

    const double total_mib = ((double) total_size) * (double) n_stream / (double) n_experts / 1024.0 / 1024.0;
    fprintf(stderr,
            "DSTORAGE_STREAM_EXPERTS layer=%d n_experts=%d warm_pool=%d gpu_mib=%d pinned_mib=%d ok=%d direct=%d slots=%d bytes=%.2f MiB elapsed=%.3f ms throughput=%.2f MiB/s pools=%zu ids=%zu\n",
            layer,
            n_stream,
            warm_pool_first ? 1 : 0,
            gpu_cache_mib,
            pinned_cache_mib,
            ok ? 1 : 0,
            direct_execution ? 1 : 0,
            slots.get_n_slots(),
            total_mib,
            elapsed_us / 1000.0,
            elapsed_us > 0 ? total_mib * 1000000.0 / elapsed_us : 0.0,
            pool_ptrs.size(),
            id_map.size());

    if (ok && repeat_load) {
        std::vector<int32_t> evict_ids(n_stream);
        for (int i = 0; i < n_stream; ++i) {
            evict_ids[i] = (n_stream + i) % n_experts;
        }
        std::unordered_map<std::string, uint64_t> evict_pool_ptrs;
        std::vector<int32_t> evict_id_map;
        const bool evict_ok = slots.ensure_experts_loaded(layer, evict_ids.data(), int(evict_ids.size()), evict_pool_ptrs, evict_id_map);
        fprintf(stderr, "DSTORAGE_STREAM_EVICT layer=%d n_experts=%d ok=%d ids=%zu\n",
                layer, n_stream, evict_ok ? 1 : 0, evict_id_map.size());

        std::unordered_map<std::string, uint64_t> pool_ptrs2;
        std::vector<int32_t> id_map2;
        const int64_t t1 = ggml_time_us();
        const bool ok2 = slots.ensure_experts_loaded(layer, expert_ids.data(), int(expert_ids.size()), pool_ptrs2, id_map2);
        const int64_t elapsed2_us = ggml_time_us() - t1;
        fprintf(stderr,
                "DSTORAGE_STREAM_EXPERTS_REPEAT layer=%d n_experts=%d pinned_mib=%d ok=%d slots=%d bytes=%.2f MiB elapsed=%.3f ms throughput=%.2f MiB/s pools=%zu ids=%zu\n",
                layer,
                n_stream,
                pinned_cache_mib,
                ok2 ? 1 : 0,
                slots.get_n_slots(),
                total_mib,
                elapsed2_us / 1000.0,
                elapsed2_us > 0 ? total_mib * 1000000.0 / elapsed2_us : 0.0,
                pool_ptrs2.size(),
                id_map2.size());
    }

    slots.destroy();
    gguf_free(ctx);
    return ok ? 0 : 1;
}
