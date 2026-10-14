"""Unit tests for benchmark parsing, aggregation and equivalence helpers."""
import importlib.util
from pathlib import Path
import unittest
import socket
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench-prerouter.py"
SPEC = importlib.util.spec_from_file_location("bench_prerouter", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def sample(arm, mode, repeat=0, prompt_id=0, content="same", tokens=(1, 2), predicted_ms=1000):
    return {
        "arm": arm,
        "mode": mode,
        "repeat": repeat,
        "prompt_id": prompt_id,
        "wall_s": predicted_ms / 1000 + 0.25,
        "response": {
            "content": content,
            "tokens": list(tokens),
            "tokens_predicted": len(tokens),
            "timings": {"predicted_ms": predicted_ms, "predicted_n": len(tokens)},
        },
    }


class BenchPrerouterTests(unittest.TestCase):
    def test_cgroup_path_is_bounded_to_unified_hierarchy(self):
        with mock.patch.object(Path, "read_text", return_value="0::/user.slice/bench.scope\n"):
            self.assertEqual(bench._cgroup_v2_path(123),
                             Path("/sys/fs/cgroup/user.slice/bench.scope"))
        with mock.patch.object(Path, "read_text", return_value="0::/../../outside\n"):
            with self.assertRaises(ValueError):
                bench._cgroup_v2_path(123)
        with mock.patch.object(Path, "read_text", return_value="2:cpu:/legacy\n"):
            self.assertIsNone(bench._cgroup_v2_path(123))

    def test_resource_value_parsing(self):
        self.assertEqual(bench._integer_key_values("max 17\noom 0\n"), {"max": 17, "oom": 0})
        self.assertEqual(bench._nvidia_value(" 80.5 "), 80.5)
        self.assertEqual(bench._nvidia_value("[N/A]"), "[N/A]")
        self.assertEqual(bench._nvidia_value("max"), "max")

    def test_port_probe_rejects_live_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaises(RuntimeError):
                bench.assert_port_available(port)
        bench.assert_port_available(port)

    def test_defaults_and_baseline_only_arguments(self):
        parser = bench.build_parser()
        args = parser.parse_args(["--server", "server", "--model", "model", "--output", "out",
                                  "--modes", "baseline"])
        modes = bench.validate_args(args, parser)
        self.assertEqual(modes, ["baseline"])
        self.assertEqual((args.speculative_slots, args.context, args.batch, args.ubatch, args.threads),
                         (16, 2048, 128, 128, 4))

    def test_completion_rejects_empty_or_invalid_tokens_and_timings(self):
        valid = sample(0, "baseline")["response"]
        self.assertEqual(bench.validate_completion(valid, 2), {"tokens": 2, "decode_s": 1.0})
        invalid_responses = []
        empty = dict(valid, tokens=[], tokens_predicted=0)
        invalid_responses.append((empty, 2))
        wrong_ids = dict(valid, tokens=[1, -1])
        invalid_responses.append((wrong_ids, 2))
        nan_timing = dict(valid, timings={"predicted_ms": float("nan"), "predicted_n": 2})
        invalid_responses.append((nan_timing, 2))
        zero_timing = dict(valid, timings={"predicted_ms": 0, "predicted_n": 2})
        invalid_responses.append((zero_timing, 2))
        for response, expected in invalid_responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                bench.validate_completion(response, expected)

    def test_throughput_uses_summed_tokens_and_elapsed_decode_time(self):
        rows = [sample(0, "baseline", tokens=(1, 2), predicted_ms=1000),
                sample(1, "baseline", prompt_id=1, tokens=(1, 2, 3, 4), predicted_ms=4000)]
        summary = bench.summarize_records(rows, mode="baseline")
        self.assertEqual(summary["tokens_predicted_total"], 6)
        self.assertEqual(summary["decode_elapsed_s_total"], 5.0)
        self.assertAlmostEqual(summary["decode_tps_total"], 1.2)
        self.assertEqual(summary["median_sample_decode_tps"], 1.5)

    def test_pairs_with_nearest_baselines_before_and_after(self):
        rows = [
            sample(0, "baseline", predicted_ms=2000),
            sample(1, "shadow", predicted_ms=1000),
            sample(2, "baseline", predicted_ms=1000),
            sample(3, "baseline", repeat=1, predicted_ms=6000),
        ]
        summary = bench.summarize_records(rows, mode="shadow")
        paired = summary["paired_comparisons"][0]
        self.assertEqual(paired["baseline_before"]["arm"], 0)
        self.assertEqual(paired["baseline_after"]["arm"], 2)
        self.assertAlmostEqual(paired["delta_before_pct"], 100.0)
        self.assertAlmostEqual(paired["delta_after_pct"], 0.0)
        self.assertAlmostEqual(paired["paired_delta_pct"], 50.0)
        self.assertAlmostEqual(summary["paired_delta_pct_total"], 50.0)

    def test_equivalence_checks_every_matching_baseline_sample(self):
        rows = [sample(0, "baseline", content="first"),
                sample(1, "shadow", content="first"),
                sample(2, "baseline", content="second")]
        result = bench.equivalence_report(rows)
        candidate = next(row for row in result["comparisons"] if row["arm"] == 1)
        self.assertEqual(candidate["baseline_arms_compared"], [0, 2])
        candidate_mismatches = [row for row in result["mismatches"] if row["arm"] == 1]
        self.assertEqual([row["baseline_arm"] for row in candidate_mismatches], [2])
        self.assertEqual(result["comparison_count"], 6)

    def test_baseline_only_has_no_paired_delta(self):
        summary = bench.summarize_records([sample(0, "baseline")], mode="baseline")
        self.assertEqual(summary["paired_sample_count"], 0)
        self.assertIsNone(summary["paired_delta_pct_total"])
        self.assertEqual(bench.equivalence_report([sample(0, "baseline")])["mismatches"], [])


if __name__ == "__main__":
    unittest.main()
