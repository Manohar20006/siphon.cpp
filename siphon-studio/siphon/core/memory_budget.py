"""Descriptor-derived tensor reservations, not a guarantee of total runtime usage."""

import re
from typing import Any

MIB = 1024 * 1024
def memory_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Graph/placement inputs. Never reuse a workspace for a different shape."""
    defaults = {"gpu_layers": 999, "context": 0, "batch": 512, "microbatch": 128,
                "parallel": 1, "draft_tokens": 0, "expert_k": 0}
    result = {key: int(config.get(key, default)) for key, default in defaults.items()}
    result.update({"vram_cache": int(config.get("vram_cache", config.get("l1_cache", 0))),
                   "cache_type": config.get("cache_type", "f16"),
                   "draft_cache_type": config.get("draft_cache_type", "f16"),
                   "spec_type": config.get("spec_type", "none"),
                   "draft_model": config.get("draft_model", ""),
                   "flash_attention": bool(config.get("flash_attention", True)),
                   "rope_scaling": config.get("rope_scaling", "auto")})
    if result["draft_model"] in {"Native MTP / NextN", "Disabled"}:
        if result["draft_model"] == "Disabled":
            result["spec_type"] = "none"
        result["draft_model"] = ""
    return result


def memory_layout(fields: dict[str, Any], tensors: list[tuple[str, int]]) -> dict[str, Any] | None:
    """Read shapes/bytes only. PLE lookup and input embeddings stay CPU-backed."""
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

    arch = str(fields.get("general.architecture", ""))
    def get(key, default=0):
        return fields.get(f"{arch}.{key}", default)
    layers = int(get("block_count"))
    heads = get("attention.head_count_kv")
    embedding = int(get("embedding_length"))
    q_heads = get("attention.head_count")
    if not layers or not heads or not embedding or not q_heads:
        return None
    hybrid = arch in {"qwen35", "qwen35moe", "qwen3next", "qwen4exp"}
    # Other recurrent, MLA, shared-KV, or SWA layouts need their own memory
    # rules. Do not label a standard-attention approximation as exact for them.
    if (get("ssm.state_size") and not hybrid) or get("attention.kv_lora_rank") or get("attention.sliding_window"):
        return None
    nextn = int(get("nextn_predict_layers"))
    main_layers = layers - nextn
    def at(value, i):
        return int(value[i] if isinstance(value, list) else value)
    recurrent = get("attention.recurrent_layers", None)
    interval = int(get("full_attention_interval", 4))
    def is_recurrent(i):
        return hybrid and i < main_layers and (
            bool(at(recurrent, i)) if recurrent is not None else (i + 1) % interval != 0
        )
    k_dim = get("attention.key_length", embedding // at(q_heads, 0))
    v_dim = get("attention.value_length", k_dim)
    layer_weights = [0] * layers
    output_weights = 0
    for name, size in tensors:
        if "exps" in name.lower() or name.endswith("per_layer_token_embd.weight"):
            continue
        match = re.match(r"blk\.(\d+)\.", name)
        if match and int(match[1]) < layers:
            layer_weights[int(match[1])] += size
        elif not name.startswith("token_embd."):
            output_weights += size
    # Tied output embeddings get a GPU view/copy even though input is CPU.
    if not any(name == "output.weight" for name, _ in tensors):
        output_weights += sum(size for name, size in tensors if name == "token_embd.weight")
    cache_types = {"f32", "f16", "bf16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "tq1_0", "tq2_0"}
    types = {kind.name.lower(): size for kind, size in GGML_QUANT_SIZES.items() if kind.name.lower() in cache_types}
    types["tq3_0"] = (32, 14)  # sizeof(block_tq3_0), including fp16 scale
    kv, draft = {}, {}
    for kind, (block, size) in types.items():
        main_rows, draft_rows = [], []
        for i in range(layers):
            k = at(heads, i) * at(k_dim, i)
            v = at(heads, i) * at(v_dim, i)
            if k % block or v % block:
                break
            row = (k + v) // block * size
            if arch == "qwen4exp" and not is_recurrent(i):
                idx_k, idx_v = int(get("attention.indexer.key_length")), at(v_dim, i)
                if not idx_k or idx_k % block or idx_v % block:
                    break
                row += (idx_k + idx_v) // block * size
            main_rows.append(row if i < main_layers and not is_recurrent(i) else 0)
            draft_rows.append(row if i >= main_layers else 0)
        else:
            kv[kind], draft[kind] = main_rows, draft_rows
    if not kv:
        return None
    conv = max(0, int(get("ssm.conv_kernel")) - 1) * (
        int(get("ssm.inner_size")) + 2 * int(get("ssm.group_count")) * int(get("ssm.state_size"))
    )
    state = int(get("ssm.inner_size")) * int(get("ssm.state_size"))
    fixed = [(conv + state) * 4 if is_recurrent(i) else 0 for i in range(layers)]
    for i in get("ple.layers", []):
        if fixed[int(i)]:
            fixed[int(i)] += max(0, int(get("ple.conv_kernel")) - 1) * int(get("ple.ngram_size")) * int(get("hyper_connection.count", 1)) * embedding * 4
    return {
        "layer_weights_bytes": layer_weights, "output_weights_bytes": output_weights,
        "kv_bytes_by_type": kv, "draft_kv_bytes_by_type": draft,
        "recurrent_bytes_by_layer": fixed,
        "rollback_supported": arch in {"qwen35", "qwen35moe"},
    }


def gpu_reservation(layout: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Follow the engine's layer placement and separate MTP cache type."""
    layers = len(layout["layer_weights_bytes"])
    ngl = max(0, int(config.get("gpu_layers", 999)))
    start = max(0, layers + 1 - ngl)
    context = max(0, int(config.get("context", 0)))
    kind = str(config.get("cache_type", "f16"))
    if kind not in layout["kv_bytes_by_type"]:
        raise ValueError(f"KV memory sizing unavailable for cache type {kind}")
    kv_bytes = sum(layout["kv_bytes_by_type"][kind][start:])
    signature = memory_signature(config)
    mtp = signature["spec_type"] == "draft-mtp" and not signature["draft_model"]
    draft_kind = str(config.get("draft_cache_type", "f16"))
    draft_bytes = sum(layout["draft_kv_bytes_by_type"][draft_kind][start:]) if mtp else 0
    rollback = max(0, int(config.get("draft_tokens", 0))) if layout.get("rollback_supported", True) and config.get("spec_type") in {"draft-mtp", "draft-dflash"} else 0
    weights = sum(layout["layer_weights_bytes"][start:]) + (layout["output_weights_bytes"] if ngl else 0)
    result = {
        "weights_mib": weights / MIB,
        "kv_bytes_per_token": kv_bytes, "kv_mib": context * kv_bytes / MIB,
        "draft_kv_bytes_per_token": draft_bytes, "draft_kv_mib": context * draft_bytes / MIB,
        "recurrent_mib": sum(layout["recurrent_bytes_by_layer"][start:]) / MIB * max(1, int(config.get("parallel", 1))) * (1 + rollback),
        # Descriptor bytes do not tell us graph workspace size. An old cached
        # compute_reserve_mib is deliberately ignored, not treated as telemetry.
        "compute_mib": 0, "workspace_measured": False,
        "expert_pool_saving_mib": 0,
    }
    result["total_mib"] = sum(result[key] for key in ("weights_mib", "kv_mib", "draft_kv_mib", "recurrent_mib", "compute_mib"))
    measured = layout.get("runtime_memory", {})
    if (measured.get("config") == signature and
            all(type(measured.get(key)) is int and measured[key] >= 0
                for key in ("model_bytes", "context_bytes", "compute_bytes"))):
        result.update(weights_mib=measured["model_bytes"] / MIB,
                      compute_mib=measured["compute_bytes"] / MIB, workspace_measured=True)
        result["recurrent_mib"] = max(0, measured["context_bytes"] / MIB - result["kv_mib"] - result["draft_kv_mib"])
        result["total_mib"] = sum(measured[key] for key in ("model_bytes", "context_bytes", "compute_bytes")) / MIB
        pool = measured.get("expert_pool_bytes")
        requested_pool = signature["vram_cache"] * MIB
        if type(pool) is int and 0 < pool <= requested_pool:
            result["expert_pool_saving_mib"] = (requested_pool - pool) / MIB
            result["total_mib"] = max(0, result["total_mib"] - result["expert_pool_saving_mib"])
    return result
