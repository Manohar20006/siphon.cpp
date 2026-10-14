"""Model-derived VRAM reservations; no engine launch or weight loading needed."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "siphon-studio"), str(ROOT / "gguf-py")]
try:
    from siphon.core.memory_budget import memory_layout, gpu_reservation
except ModuleNotFoundError:
    memory_layout = gpu_reservation = None

MIB = 1024 * 1024


def qwen_fields():
    return {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 41,
        "qwen35moe.nextn_predict_layers": 1,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.ssm.inner_size": 4096,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.group_count": 16,
    }


class MemoryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(memory_layout, "model-derived budget implementation missing")
        self.layout = memory_layout(qwen_fields(), [
            ("token_embd.weight", 512 * MIB),
            ("output.weight", 512 * MIB),
            ("blk.0.attn_norm.weight", MIB),
            ("blk.39.attn_k.weight", 2 * MIB),
            ("blk.40.nextn.eh_proj.weight", MIB),
            ("blk.0.ffn_up_exps.weight", 2048 * MIB),
        ])

    def test_main_tq3_kv_uses_ten_attention_layers_and_block_scales(self):
        # 10 layers * (K+V) * 512 values * 14 bytes / 32 values.
        self.assertEqual(sum(self.layout["kv_bytes_by_type"]["tq3_0"]), 4480)
        self.assertEqual(sum(self.layout["kv_bytes_by_type"]["f16"]), 20480)

    def test_unmeasured_workspace_is_not_a_fabricated_512_mib_allocation(self):
        result = gpu_reservation(self.layout, {"context": 65536})
        self.assertEqual(result["compute_mib"], 0)
        self.assertFalse(result["workspace_measured"])

    def test_measured_buffers_allow_the_previously_working_2816_mib_pool(self):
        from types import SimpleNamespace
        from siphon.core.memory_budget import memory_signature
        from siphon.desktop.app import _resource_state, _validate_resource_settings
        config = {"gpu_layers": 999, "context": 65536, "cache_type": "tq3_0",
                  "spec_type": "draft-mtp", "draft_tokens": 3, "l1_cache": 2816}
        self.layout["runtime_memory"] = {
            "config": memory_signature(config), "gpu_name": "test GPU",
            "model_bytes": 2135327232, "context_bytes": 691273728,
            "compute_bytes": 146001152,
            # Logical budget is a ceiling; mixed-stride slot groups allocate
            # less than that ceiling. Do not charge the unallocated remainder.
            "expert_pool_bytes": 2755490304,
        }
        system = SimpleNamespace(mem_free_gb=10, mem_total_gb=16, gpu_mib_free=5759,
                                 gpu_mib_total=6141, gpu_name="test GPU")
        state = _resource_state(system, config, self.layout)
        self.assertGreaterEqual(state["vram_safe_mib"], 2816)
        self.assertTrue(state["vram_reservation"]["workspace_measured"])
        self.assertTrue(_validate_resource_settings(config, system, memory_layout=self.layout)[0])
        # A measurement is not reusable for a different GPU or graph shape.
        system.gpu_name = "other GPU"
        self.assertFalse(_resource_state(system, config, self.layout)["vram_reservation"]["workspace_measured"])
        changed = gpu_reservation(self.layout, {**config, "microbatch": 256})
        self.assertFalse(changed["workspace_measured"])
        # Apply sends a friendly option name; it is the same native-MTP shape.
        same = gpu_reservation(self.layout, {**config, "draft_model": "Native MTP / NextN"})
        self.assertTrue(same["workspace_measured"])

    def test_mtp_uses_separate_f16_cache_not_the_targets_tq3_type(self):
        result = gpu_reservation(self.layout, {
            "gpu_layers": 999, "context": 65536, "cache_type": "tq3_0",
            "spec_type": "draft-mtp", "draft_tokens": 3,
        })
        self.assertEqual(result["kv_bytes_per_token"], 4480)
        self.assertEqual(result["draft_kv_bytes_per_token"], 2048)
        self.assertEqual(result["kv_mib"], 280)
        self.assertEqual(result["draft_kv_mib"], 128)
        # 30 recurrent layers * (24576 conv + 524288 state) * f32 * 4 rows.
        self.assertEqual(result["recurrent_mib"], 251.25)

    def test_cpu_input_and_streamed_experts_are_not_reserved_as_gpu_weights(self):
        result = gpu_reservation(self.layout, {"gpu_layers": 999, "context": 4096})
        self.assertEqual(result["weights_mib"], 516)

    def test_partial_placement_matches_engine_output_layer_slot(self):
        # Engine routes slots [layers + 1 - ngl, layers], including output.
        result = gpu_reservation(self.layout, {"gpu_layers": 3, "context": 4096})
        self.assertEqual(result["weights_mib"], 515)
        self.assertEqual(result["kv_bytes_per_token"], 2048)  # F16 on layer 39
        self.assertEqual(gpu_reservation(self.layout, {"gpu_layers": 0})["total_mib"], 0)

    def test_ple_lookup_is_excluded_and_indexer_cache_is_included(self):
        fields = qwen_fields()
        fields = {key.replace("qwen35moe", "qwen4exp"): value for key, value in fields.items()}
        fields["general.architecture"] = "qwen4exp"
        fields["qwen4exp.nextn_predict_layers"] = 0
        fields["qwen4exp.attention.indexer.key_length"] = 128
        layout = memory_layout(fields, [
            ("per_layer_token_embd.weight", 27 * 1024 * MIB),
            ("output.weight", 100 * MIB),
        ])
        result = gpu_reservation(layout, {"gpu_layers": 999, "context": 4096, "cache_type": "tq3_0"})
        self.assertEqual(result["weights_mib"], 100)
        # 10 attention layers, indexer K=128 and V=256, both use TQ3.
        self.assertEqual(result["kv_bytes_per_token"], 6160)

    def test_missing_kv_dimensions_does_not_fabricate_an_exact_budget(self):
        self.assertIsNone(memory_layout({"general.architecture": "unknown", "unknown.block_count": 2}, []))

    def test_loaded_model_reservations_are_not_subtracted_twice(self):
        from types import SimpleNamespace
        from siphon.desktop.app import _resource_state
        system = SimpleNamespace(mem_free_gb=10, gpu_mib_free=4096)
        config = {"gpu_layers": 999, "context": 65536, "cache_type": "tq3_0", "spec_type": "draft-mtp", "draft_tokens": 3}
        offline = _resource_state(system, config, self.layout)
        loaded = _resource_state(system, config, self.layout, running=True)
        self.assertEqual(offline["vram_safe_mib"], 2792.75)
        self.assertEqual(loaded["vram_safe_mib"], 3968)

    def test_unknown_model_memory_is_rejected_when_required_for_launch(self):
        from types import SimpleNamespace
        from siphon.desktop.app import _validate_resource_settings
        system = SimpleNamespace(mem_free_gb=16, mem_total_gb=16, gpu_mib_free=8192, gpu_mib_total=8192)
        ok, reason = _validate_resource_settings({}, system, require_model_budget=True)
        self.assertFalse(ok)
        self.assertIn("memory", reason.lower())

    def test_unsupported_cache_type_does_not_interrupt_hardware_polling(self):
        from types import SimpleNamespace
        from siphon.desktop.app import _resource_state, _validate_resource_settings
        system = SimpleNamespace(mem_free_gb=16, mem_total_gb=16, gpu_mib_free=8192, gpu_mib_total=8192)
        config = {"cache_type": "unsupported", "context": 4096}
        state = _resource_state(system, config, self.layout)
        self.assertFalse(state["vram_budget_known"])
        self.assertEqual(state["vram_safe_mib"], 0)
        self.assertFalse(_validate_resource_settings(config, system, memory_layout=self.layout)[0])

    def test_model_buffers_must_fit_even_when_expert_cache_is_zero(self):
        from types import SimpleNamespace
        from siphon.desktop.app import _validate_resource_settings
        system = SimpleNamespace(mem_free_gb=16, mem_total_gb=16, gpu_mib_free=1024, gpu_mib_total=6144)
        ok, reason = _validate_resource_settings({"context": 65536, "vram_cache": 0}, system, memory_layout=self.layout)
        self.assertFalse(ok)
        self.assertIn("model", reason.lower())

    def test_inspection_exposes_layout_for_a_real_gguf_fixture(self):
        import tempfile
        import numpy as np
        from gguf import GGUFWriter
        from siphon.core.model_files import inspect_model_files
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.gguf"
            writer = GGUFWriter(str(path), "qwen35moe")
            for key, value in qwen_fields().items():
                if key != "general.architecture":
                    writer.add_uint32(key, value)
            writer.add_expert_count(4)
            writer.add_expert_used_count(2)
            writer.add_tensor("blk.0.ffn_up_exps.weight", np.ones((2, 2, 2), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            info = inspect_model_files(path)
        self.assertEqual(sum(info.get("memory_layout", {}).get("kv_bytes_by_type", {}).get("tq3_0", [])), 4480)


if __name__ == "__main__":
    unittest.main()
