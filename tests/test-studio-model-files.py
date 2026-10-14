"""Tests for GGUF folder discovery using small files from the bundled writer."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "siphon-studio"))
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf.gguf_writer import GGUFWriter
try:
    from siphon.core.model_files import (
        inspect_model_files,
        model_fingerprint,
        model_part_paths,
        scan_model_folder,
    )
except ModuleNotFoundError as exc:
    if exc.name != "siphon.core.model_files":
        raise
    inspect_model_files = None
    model_fingerprint = None
    model_part_paths = None
    scan_model_folder = None


def write_model(
    folder: Path,
    stem: str,
    *,
    part_count: int = 1,
    part_indices: range | tuple[int, ...] | None = None,
    name: str = "Fixture MoE",
    architecture: str = "qwen2",
    expert_count: int | None = 4,
    expert_used_count: int | None = 2,
    block_count: int | None = 2,
    context_length: int | None = 4096,
    tensor_names: list[str] | None = None,
    tensor_names_by_part: list[list[str]] | None = None,
    optional_metadata_by_part: dict[int, dict[str, str]] | None = None,
    declared_tensor_count: int | None = None,
    declared_split_count: int | None = None,
    declared_split_no: dict[int, int] | None = None,
    general_type: str | None = "model",
) -> list[Path]:
    """Write real GGUF files, including standard split metadata when requested."""
    folder.mkdir(parents=True, exist_ok=True)
    indices = tuple(part_indices if part_indices is not None else range(part_count))
    if tensor_names is None:
        tensor_names = [
            f"blk.{index}.ffn_up_exps.weight" if expert_count else "token_embd.weight"
            for index in range(part_count)
        ]
    if tensor_names_by_part is None:
        tensor_names_by_part = [
            [tensor_names[min(index, len(tensor_names) - 1)]]
            for index in range(part_count)
        ]
    if len(tensor_names_by_part) != part_count:
        raise ValueError("tensor_names_by_part must contain one group per shard")
    total_tensors = (
        declared_tensor_count
        if declared_tensor_count is not None
        else sum(len(names) for names in tensor_names_by_part)
    )
    paths = []

    for index in indices:
        if part_count == 1:
            path = folder / f"{stem}.gguf"
        else:
            path = folder / f"{stem}-{index + 1:05d}-of-{part_count:05d}.gguf"
        writer = GGUFWriter(path, arch=architecture)
        writer.add_name(name)
        if general_type:
            writer.add_type(general_type)
        if block_count is not None:
            writer.add_block_count(block_count)
        if context_length is not None:
            writer.add_context_length(context_length)
        if expert_count is not None:
            writer.add_expert_count(expert_count)
        if expert_used_count is not None:
            writer.add_expert_used_count(expert_used_count)
        if part_count > 1:
            writer.add_uint16("split.no", (declared_split_no or {}).get(index, index))
            writer.add_uint16("split.count", declared_split_count or part_count)
            writer.add_int32("split.tensors.count", total_tensors)
        for key, value in (optional_metadata_by_part or {}).get(index, {}).items():
            writer.add_string(key, value)
        for tensor_name in tensor_names_by_part[index]:
            shape = (2, 2, 2) if "exps" in tensor_name else (4, 4)
            writer.add_tensor(tensor_name, np.ones(shape, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(path)
    return paths


class ModelFilesTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            scan_model_folder,
            "siphon.core.model_files is missing the Task 1 API",
        )
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def entry(self, entries, name):
        return next(item for item in entries if item["name"] == name)

    def test_inspects_single_moe_and_reports_only_declared_metadata(self):
        (path,) = write_model(
            self.folder, "single", name="Single MoE", block_count=None,
            context_length=None,
        )

        metadata = inspect_model_files(path)

        self.assertEqual(metadata["name"], "Single MoE")
        self.assertEqual(metadata["architecture"], "qwen2")
        self.assertIsNone(metadata["layers"])
        self.assertIsNone(metadata["native_context"])
        self.assertEqual(metadata["expert_count"], 4)
        self.assertEqual(metadata["expert_used_count"], 2)
        self.assertEqual(metadata["part_count"], 1)
        self.assertEqual(metadata["parts"], [str(path.resolve())])
        self.assertEqual(metadata["expert_gib"], 32 / (1024 ** 3))
        self.assertEqual(metadata["dense_gib"], 0.0)

    def test_tensor_breakdown_separates_auxiliary_lookup_from_legacy_dense_total(self):
        (path,) = write_model(
            self.folder,
            "lookup",
            tensor_names_by_part=[[
                "per_layer_token_embd.weight",
                "token_embd.weight",
                "blk.0.attn_q.weight",
                "blk.0.ffn_up_exps.weight",
            ]],
        )

        metadata = inspect_model_files(path)

        self.assertEqual(metadata["dense_gib"], 192 / (1024 ** 3))
        self.assertEqual(metadata["core_gib"], 128 / (1024 ** 3))
        self.assertEqual(metadata["lookup_gib"], 64 / (1024 ** 3))
        self.assertEqual(metadata["expert_gib"], 32 / (1024 ** 3))
        self.assertEqual(metadata["tensor_gib"], 224 / (1024 ** 3))
        self.assertAlmostEqual(
            metadata["core_gib"] + metadata["lookup_gib"] + metadata["expert_gib"],
            metadata["tensor_gib"],
        )

    def test_split_breakdown_counts_lookup_table_on_a_later_part(self):
        paths = write_model(
            self.folder,
            "split-lookup",
            part_count=2,
            tensor_names_by_part=[
                ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_up_exps.weight"],
                ["model.per_layer_token_embd.weight"],
            ],
        )

        metadata = inspect_model_files(paths[0])

        self.assertEqual(metadata["dense_gib"], 192 / (1024 ** 3))
        self.assertEqual(metadata["core_gib"], 128 / (1024 ** 3))
        self.assertEqual(metadata["lookup_gib"], 64 / (1024 ** 3))
        self.assertEqual(metadata["expert_gib"], 32 / (1024 ** 3))
        self.assertEqual(metadata["tensor_gib"], 224 / (1024 ** 3))

    def test_lookup_classification_requires_exact_namespaced_suffix(self):
        (path,) = write_model(
            self.folder,
            "lookup-names",
            tensor_names_by_part=[[
                "language_model.per_layer_token_embd.weight",
                "per_layer_token_embd.weight_extra",
                "not_per_layer_token_embd.weight",
                "per_layer_token_embd.weight.extra",
                "token_embd.weight",
                "blk.0.attn_q.weight",
                "blk.0.ffn_up_exps.weight",
            ]],
        )

        metadata = inspect_model_files(path)

        self.assertEqual(metadata["lookup_gib"], 64 / (1024 ** 3))
        self.assertEqual(metadata["core_gib"], 5 * 64 / (1024 ** 3))
        self.assertEqual(metadata["tensor_gib"], 416 / (1024 ** 3))

    def test_known_absent_lookup_is_zero_and_incomplete_preview_stays_unknown(self):
        (dense_path,) = write_model(
            self.folder,
            "without-lookup",
            tensor_names_by_part=[["token_embd.weight", "blk.0.ffn_up_exps.weight"]],
        )
        split = write_model(
            self.folder,
            "incomplete-lookup",
            part_count=2,
            name="Incomplete Lookup",
            tensor_names_by_part=[
                ["blk.0.ffn_up_exps.weight"],
                ["per_layer_token_embd.weight"],
            ],
        )
        split[1].unlink()

        metadata = inspect_model_files(dense_path)
        incomplete = self.entry(scan_model_folder(self.folder), "Incomplete Lookup")["metadata"]

        self.assertEqual(metadata["core_gib"], 64 / (1024 ** 3))
        self.assertEqual(metadata["lookup_gib"], 0.0)
        self.assertEqual(metadata["expert_gib"], 32 / (1024 ** 3))
        self.assertEqual(metadata["tensor_gib"], 96 / (1024 ** 3))
        self.assertIsNone(incomplete["core_gib"])
        self.assertIsNone(incomplete["lookup_gib"])
        self.assertIsNone(incomplete["tensor_gib"])

    def test_complete_invalid_preview_still_aggregates_descriptor_breakdown(self):
        write_model(
            self.folder,
            "dense-preview",
            name="Dense Preview",
            expert_count=None,
            expert_used_count=None,
            tensor_names_by_part=[[
                "per_layer_token_embd.weight",
                "token_embd.weight",
            ]],
        )

        entry = self.entry(scan_model_folder(self.folder), "Dense Preview")

        self.assertFalse(entry["available"])
        self.assertEqual(entry["metadata"]["dense_gib"], 128 / (1024 ** 3))
        self.assertEqual(entry["metadata"]["core_gib"], 64 / (1024 ** 3))
        self.assertEqual(entry["metadata"]["lookup_gib"], 64 / (1024 ** 3))
        self.assertEqual(entry["metadata"]["expert_gib"], 0.0)
        self.assertEqual(entry["metadata"]["tensor_gib"], 128 / (1024 ** 3))

    def test_accepts_five_and_eight_shards_and_aggregates_real_tensor_bytes(self):
        five = write_model(self.folder, "five", part_count=5, name="Five MoE")
        eight = write_model(self.folder, "eight", part_count=8, name="Eight MoE")

        five_metadata = inspect_model_files(five[0])
        eight_metadata = inspect_model_files(eight[0])

        self.assertEqual(five_metadata["part_count"], 5)
        self.assertEqual(eight_metadata["part_count"], 8)
        self.assertEqual(len(five_metadata["parts"]), 5)
        self.assertEqual(len(eight_metadata["parts"]), 8)
        self.assertAlmostEqual(five_metadata["expert_gib"], 5 * 32 / (1024 ** 3))
        self.assertAlmostEqual(eight_metadata["expert_gib"], 8 * 32 / (1024 ** 3))
        self.assertAlmostEqual(
            five_metadata["artifact_gib"],
            sum(path.stat().st_size for path in five) / (1024 ** 3),
        )

    def test_split_identity_ignores_reader_header_counts_for_uneven_parts(self):
        paths = write_model(
            self.folder,
            "uneven",
            part_count=2,
            name="Uneven MoE",
            tensor_names_by_part=[
                ["blk.0.ffn_up_exps.weight", "blk.1.ffn_up_exps.weight"],
                ["blk.2.ffn_up_exps.weight"],
            ],
            optional_metadata_by_part={1: {"fixture.optional": "second shard"}},
        )

        metadata = inspect_model_files(paths[0])

        self.assertEqual(metadata["part_count"], 2)
        self.assertAlmostEqual(metadata["expert_gib"], 3 * 32 / (1024 ** 3))

    def test_standard_part_paths_are_ordered_even_when_some_are_missing(self):
        parts = model_part_paths(self.folder / "family-00003-of-00005.gguf")

        self.assertEqual(
            parts,
            [self.folder / f"family-{index:05d}-of-00005.gguf" for index in range(1, 6)],
        )

    def test_invalid_split_filename_is_visible_without_hiding_valid_neighbors(self):
        (invalid_path,) = write_model(self.folder, "invalid-total", name="Invalid Total")
        invalid_path.rename(self.folder / "invalid-00001-of-00000.gguf")
        write_model(self.folder, "valid", name="Valid Neighbor")

        entries = scan_model_folder(self.folder)

        invalid = self.entry(entries, "Invalid Total")
        self.assertFalse(invalid["available"])
        self.assertIn("invalid standard split filename", invalid["error"].lower())
        self.assertTrue(self.entry(entries, "Valid Neighbor")["available"])

    def test_fingerprint_preserves_part_order_and_uses_canonical_stat_values(self):
        paths = write_model(self.folder, "fingerprint", part_count=2)

        fingerprint = model_fingerprint([paths[1], paths[0]])

        self.assertEqual([item["path"] for item in fingerprint], [
            str(paths[1].resolve()), str(paths[0].resolve()),
        ])
        self.assertEqual([item["size"] for item in fingerprint], [
            paths[1].stat().st_size, paths[0].stat().st_size,
        ])
        self.assertEqual([item["mtime_ns"] for item in fingerprint], [
            paths[1].stat().st_mtime_ns, paths[0].stat().st_mtime_ns,
        ])

    def test_scan_returns_one_entry_for_a_complete_split_model(self):
        parts = write_model(self.folder, "folder-model", part_count=3, name="Folder MoE")

        entries = scan_model_folder(self.folder)

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertTrue(entry["available"], entry["error"])
        self.assertEqual(entry["name"], "Folder MoE")
        self.assertEqual(entry["parts"], [str(path.resolve()) for path in parts])
        self.assertEqual(entry["path"], str(parts[0].resolve()))
        self.assertEqual(entry["part_count"], 3)
        self.assertEqual(entry["error"], "")

    def test_scan_reports_duplicate_numeric_shard_indices_as_ambiguous(self):
        parts = write_model(self.folder, "ambiguous", part_count=2, name="Ambiguous Set")
        duplicate = self.folder / "ambiguous-000001-of-00002.gguf"
        duplicate.write_bytes(parts[0].read_bytes())
        write_model(self.folder, "independent", name="Independent Model")

        entries = scan_model_folder(self.folder)

        ambiguous = self.entry(entries, "Ambiguous Set")
        self.assertFalse(ambiguous["available"])
        self.assertIn("ambiguous", ambiguous["error"].lower())
        self.assertEqual(
            set(ambiguous["parts"]),
            {str(path.resolve()) for path in [*parts, duplicate]},
        )
        self.assertTrue(self.entry(entries, "Independent Model")["available"])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            inspect_model_files(parts[0])

    def test_scan_keeps_same_name_quantizations_as_separate_entries(self):
        write_model(self.folder, "fixture-Q4_K_M", name="Same Display Name")
        write_model(self.folder, "fixture-Q8_0", name="Same Display Name")

        entries = scan_model_folder(self.folder)

        self.assertEqual(len(entries), 2)
        self.assertEqual(len({item["id"] for item in entries}), 2)
        self.assertNotEqual(entries[0]["path"], entries[1]["path"])

    def test_scan_ignores_non_gguf_and_keeps_independent_models_separate(self):
        write_model(self.folder, "one", name="One")
        write_model(self.folder, "two", name="Two")
        (self.folder / "notes.txt").write_text("not a model")

        entries = scan_model_folder(self.folder)

        self.assertEqual({item["name"] for item in entries}, {"One", "Two"})

    def test_each_missing_shard_is_visible_and_unavailable(self):
        for missing in range(3):
            case = self.folder / f"missing-{missing}"
            parts = write_model(case, "model", part_count=3, name=f"Missing {missing}")
            parts[missing].unlink()

            entries = scan_model_folder(case)
            item = self.entry(entries, f"Missing {missing}")

            self.assertFalse(item["available"])
            self.assertIn(f"part {missing + 1} of 3", item["error"].lower())
            self.assertEqual(item["path"], str(parts[0].resolve()))
            with self.assertRaisesRegex(ValueError, rf"part {missing + 1} of 3"):
                inspect_model_files(parts[1])

    def test_nonstandard_renamed_split_is_visible_but_cannot_be_inspected(self):
        parts = write_model(self.folder, "renamed", part_count=2)
        renamed = self.folder / "weights-part-a.gguf"
        parts[0].rename(renamed)
        parts[1].unlink()

        entries = scan_model_folder(self.folder)

        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0]["available"])
        self.assertIn("standard split filename", entries[0]["error"].lower())
        with self.assertRaisesRegex(ValueError, "standard split filename"):
            inspect_model_files(renamed)

    def test_corrupt_split_and_valid_neighbor_are_reported_independently(self):
        broken = write_model(self.folder, "broken", part_count=2, name="Broken")
        broken[1].write_bytes(b"not a GGUF")
        write_model(self.folder, "valid", name="Valid Neighbor")

        entries = scan_model_folder(self.folder)

        self.assertEqual(len(entries), 2)
        self.assertTrue(self.entry(entries, "Valid Neighbor")["available"])
        broken_entry = self.entry(entries, "Broken")
        self.assertFalse(broken_entry["available"])
        self.assertIn("unreadable", broken_entry["error"].lower())

    def test_declared_split_total_must_match_filename_and_every_part(self):
        paths = write_model(
            self.folder, "bad-total", part_count=2, name="Bad Total",
            declared_split_count=3,
        )

        with self.assertRaisesRegex(ValueError, "split.count"):
            inspect_model_files(paths[0])
        entry = self.entry(scan_model_folder(self.folder), "Bad Total")
        self.assertFalse(entry["available"])
        self.assertIn("split.count", entry["error"])

    def test_split_tensor_total_must_match_descriptors(self):
        paths = write_model(
            self.folder, "bad-tensors", part_count=2, name="Bad Tensor Total",
            declared_tensor_count=9,
        )

        with self.assertRaisesRegex(ValueError, "tensor count"):
            inspect_model_files(paths[0])

    def test_split_number_must_match_its_standard_filename_position(self):
        paths = write_model(
            self.folder, "bad-index", part_count=2, name="Bad Split Index",
            declared_split_no={1: 0},
        )

        with self.assertRaisesRegex(ValueError, "split.no"):
            inspect_model_files(paths[0])

    def test_duplicate_tensor_names_across_parts_invalidate_only_their_set(self):
        paths = write_model(
            self.folder, "duplicates", part_count=2, name="Duplicate Tensors",
            tensor_names=["blk.0.ffn_up_exps.weight"] * 2,
        )
        write_model(self.folder, "healthy", name="Healthy")

        entries = scan_model_folder(self.folder)

        duplicate = self.entry(entries, "Duplicate Tensors")
        self.assertFalse(duplicate["available"])
        self.assertIn("duplicate tensor", duplicate["error"].lower())
        self.assertTrue(self.entry(entries, "Healthy")["available"])
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            inspect_model_files(paths[0])

    def test_conflicting_declared_identity_fields_invalidate_the_split(self):
        paths = write_model(self.folder, "identity", part_count=2, name="Identity")
        paths[1].unlink()
        write_model(
            self.folder, "identity", part_count=2, part_indices=(1,),
            name="Different Identity",
        )

        with self.assertRaisesRegex(ValueError, "general.name"):
            inspect_model_files(paths[0])

    def test_dense_model_is_reported_separately_from_a_valid_moe(self):
        write_model(
            self.folder, "dense", name="Dense Model", architecture="llama",
            expert_count=None, expert_used_count=None,
        )
        write_model(self.folder, "moe", name="Routed MoE")

        entries = scan_model_folder(self.folder)

        dense = self.entry(entries, "Dense Model")
        self.assertFalse(dense["available"])
        self.assertIn("dense", dense["error"].lower())
        self.assertTrue(self.entry(entries, "Routed MoE")["available"])

    def test_auxiliary_projector_is_not_exposed_as_a_runnable_model(self):
        write_model(
            self.folder, "mmproj-model", name="Vision Projector",
            architecture="clip", expert_count=None, expert_used_count=None,
            general_type="mmproj",
        )

        entries = scan_model_folder(self.folder)

        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0]["available"])
        self.assertIn("auxiliary", entries[0]["error"].lower())

    def test_missing_architecture_is_an_explicit_error_not_a_default(self):
        path = self.folder / "bad.gguf"
        writer = GGUFWriter(path, arch="qwen2")
        writer.kv_data[0].pop("general.architecture")
        writer.add_name("No Architecture")
        writer.add_expert_count(4)
        writer.add_expert_used_count(2)
        writer.add_tensor("blk.0.ffn_up_exps.weight", np.ones((2, 2, 2), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        entry = self.entry(scan_model_folder(self.folder), "No Architecture")

        self.assertFalse(entry["available"])
        self.assertIn("architecture", entry["error"].lower())


if __name__ == "__main__":
    unittest.main()
