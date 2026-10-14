#!/usr/bin/env python3
"""Local deterministic MTP A/B runner; retains responses, timings and server logs.

No model/checkpoint downloads, OS cache flushing or modifications to profiles.
Run inside a memory-limited systemd scope on resource-constrained machines.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


RUNTIME_ENV_KEYS = (
    "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL",
)
CONTROLLED_ENV_KEYS = (
    "LLAMA_MOE_EXPERT_USED_OVERRIDE", "GGML_CUDA_DISABLE_GRAPHS",
    "LLAMA_DSTORAGE_SUMMARY", "LLAMA_DSTORAGE_PREFILL_WORKSPACE",
    "LLAMA_DSTORAGE_EVICT_TO_RAM", "LLAMA_DSTORAGE_SPECULATIVE_SLOTS",
    "LLAMA_GDS_READ_THREADS", "LLAMA_DSTORAGE_PREROUTER_PROFILE",
    "LLAMA_DSTORAGE_PREROUTER_MODE", "LLAMA_DSTORAGE_SPECULATIVE_RESERVE_ALWAYS",
)


def request(port, path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode()
    return json.loads(raw) if path != "/metrics" else raw


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", default="baseline,shadow,prefetch,baseline")
    parser.add_argument("--tokens", default=128, type=int)
    parser.add_argument("--repeats", default=2, type=int)
    parser.add_argument("--port", default=18091, type=int)
    parser.add_argument("--gpu-cache-mib", default=2800, type=int)
    parser.add_argument("--pinned-cache-mib", default=2048, type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reserve-always", action="store_true",
                        help="keep the explicit speculative reserve during all phases, in every arm")
    parser.add_argument("--speculative-slots", default=16, type=int)
    parser.add_argument("--context", default=2048, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--ubatch", default=128, type=int)
    parser.add_argument("--threads", default=4, type=int)
    return parser


def validate_args(args, parser):
    modes = [mode.strip() for mode in args.modes.split(",")]
    if not modes or any(mode not in ("baseline", "shadow", "prefetch") for mode in modes):
        parser.error("unknown mode")
    if "baseline" not in modes:
        parser.error("at least one baseline arm is required")
    if args.tokens <= 0 or args.repeats <= 0:
        parser.error("--tokens and --repeats must be positive")
    if args.gpu_cache_mib <= 0 or args.pinned_cache_mib < 0:
        parser.error("invalid cache budget")
    if args.speculative_slots <= 0:
        parser.error("--speculative-slots must be positive")
    if any(value <= 0 for value in (args.context, args.batch, args.ubatch, args.threads)):
        parser.error("--context, --batch, --ubatch and --threads must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if any(mode != "baseline" for mode in modes) and not args.checkpoint:
        parser.error("shadow/prefetch require --checkpoint")
    return modes


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_completion(response, expected_tokens):
    """Return normalized token and decode-time measurements or raise ValueError."""
    if not isinstance(expected_tokens, int) or isinstance(expected_tokens, bool) or expected_tokens <= 0:
        raise ValueError("expected token count must be a positive integer")
    if not isinstance(response, dict) or "error" in response:
        raise ValueError("completion response is missing or contains an error")
    tokens = response.get("tokens")
    if (not isinstance(tokens, list) or len(tokens) != expected_tokens or
            any(not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in tokens)):
        raise ValueError("completion token IDs are empty, invalid, or incomplete")
    predicted = response.get("tokens_predicted")
    if not isinstance(predicted, int) or isinstance(predicted, bool) or predicted != expected_tokens:
        raise ValueError("tokens_predicted does not match the requested token count")
    timings = response.get("timings")
    if not isinstance(timings, dict) or not is_finite_number(timings.get("predicted_ms")):
        raise ValueError("completion is missing valid timings.predicted_ms")
    predicted_ms = timings["predicted_ms"]
    if predicted_ms <= 0:
        raise ValueError("timings.predicted_ms must be positive")
    if "predicted_n" in timings:
        predicted_n = timings["predicted_n"]
        if (not isinstance(predicted_n, int) or isinstance(predicted_n, bool) or
                predicted_n != expected_tokens):
            raise ValueError("timings.predicted_n does not match the requested token count")
    if not isinstance(response.get("content"), str):
        raise ValueError("completion content is missing or invalid")
    return {"tokens": expected_tokens, "decode_s": predicted_ms / 1000.0}


def record_measurements(record):
    response = record.get("response", {})
    expected_tokens = response.get("tokens_predicted") if isinstance(response, dict) else None
    if not isinstance(expected_tokens, int) or isinstance(expected_tokens, bool) or expected_tokens <= 0:
        raise ValueError("record has invalid tokens_predicted")
    measured = validate_completion(response, expected_tokens)
    wall_s = record.get("wall_s")
    if not is_finite_number(wall_s) or wall_s <= 0:
        raise ValueError("record has invalid wall_s")
    return {
        "tokens": measured["tokens"],
        "decode_s": measured["decode_s"],
        "wall_s": wall_s,
        "decode_tps": measured["tokens"] / measured["decode_s"],
        "wall_tps": measured["tokens"] / wall_s,
    }


def _paired_reference(records, target):
    key = (target["repeat"], target["prompt_id"])
    candidates = [record for record in records
                  if record["mode"] == "baseline" and
                  (record["repeat"], record["prompt_id"]) == key]
    before = [record for record in candidates if record["arm"] < target["arm"]]
    after = [record for record in candidates if record["arm"] > target["arm"]]
    nearest_before = max(before, key=lambda row: row["arm"]) if before else None
    nearest_after = min(after, key=lambda row: row["arm"]) if after else None
    refs = [record for record in (nearest_before, nearest_after) if record is not None]
    if not refs:
        return None
    reference_seconds = statistics.mean(record_measurements(row)["decode_s"] for row in refs)
    target_metrics = record_measurements(target)
    paired_delta_pct = (reference_seconds / target_metrics["decode_s"] - 1.0) * 100.0

    def baseline_detail(row):
        if row is None:
            return None
        metrics = record_measurements(row)
        return {"arm": row["arm"], "decode_tps": metrics["decode_tps"],
                "decode_s": metrics["decode_s"]}

    def delta(row):
        if row is None:
            return None
        return (record_measurements(row)["decode_s"] /
                target_metrics["decode_s"] - 1.0) * 100.0

    return {
        "arm": target["arm"], "mode": target["mode"], "repeat": target["repeat"],
        "prompt_id": target["prompt_id"], "decode_tps": target_metrics["decode_tps"],
        "baseline_before": baseline_detail(nearest_before),
        "baseline_after": baseline_detail(nearest_after),
        "delta_before_pct": delta(nearest_before),
        "delta_after_pct": delta(nearest_after),
        "paired_baseline_decode_s": reference_seconds,
        "paired_delta_pct": paired_delta_pct,
    }


def summarize_records(records, mode=None, prompt_id=None):
    selected = [record for record in records
                if (mode is None or record["mode"] == mode) and
                (prompt_id is None or record["prompt_id"] == prompt_id)]
    samples = [record_measurements(record) for record in selected]
    total_tokens = sum(sample["tokens"] for sample in samples)
    total_decode_s = sum(sample["decode_s"] for sample in samples)
    total_wall_s = sum(sample["wall_s"] for sample in samples)
    paired = [_paired_reference(records, row) for row in selected if row["mode"] != "baseline"]
    paired = [item for item in paired if item is not None]
    before_deltas = [item["delta_before_pct"] for item in paired
                     if item["delta_before_pct"] is not None]
    after_deltas = [item["delta_after_pct"] for item in paired
                    if item["delta_after_pct"] is not None]
    paired_baseline_s = sum(item["paired_baseline_decode_s"] for item in paired)
    paired_tokens = sum(record_measurements(row)["tokens"] for row in selected
                        if row["mode"] != "baseline" and
                        _paired_reference(records, row) is not None)
    paired_target_s = sum(record_measurements(row)["decode_s"] for row in selected
                          if row["mode"] != "baseline" and
                          _paired_reference(records, row) is not None)
    return {
        "mode": mode,
        "prompt_id": prompt_id,
        "sample_count": len(samples),
        "tokens_predicted_total": total_tokens,
        "decode_elapsed_s_total": total_decode_s,
        "wall_elapsed_s_total": total_wall_s,
        "decode_tps_total": total_tokens / total_decode_s if total_decode_s else None,
        "wall_tps_total": total_tokens / total_wall_s if total_wall_s else None,
        "median_sample_decode_tps": statistics.median(s["decode_tps"] for s in samples) if samples else None,
        "median_sample_wall_tps": statistics.median(s["wall_tps"] for s in samples) if samples else None,
        "paired_sample_count": len(paired),
        "paired_baseline_decode_tps_total": paired_tokens / paired_baseline_s if paired_baseline_s else None,
        "paired_target_decode_tps_total": paired_tokens / paired_target_s if paired_target_s else None,
        "paired_delta_pct_total": ((paired_baseline_s / paired_target_s - 1.0) * 100.0
                                    if paired_target_s else None),
        "median_paired_delta_pct": statistics.median(item["paired_delta_pct"] for item in paired) if paired else None,
        "median_delta_before_pct": statistics.median(before_deltas) if before_deltas else None,
        "median_delta_after_pct": statistics.median(after_deltas) if after_deltas else None,
        "paired_comparisons": paired,
    }


def equivalence_report(records):
    baselines = {}
    for row in records:
        if row["mode"] == "baseline":
            baselines.setdefault((row["repeat"], row["prompt_id"]), []).append(row)
    comparisons = []
    mismatches = []
    for row in records:
        key = (row["repeat"], row["prompt_id"])
        references = baselines.get(key, [])
        if not references:
            mismatches.append({"arm": row["arm"], "mode": row["mode"],
                               "repeat": row["repeat"], "prompt_id": row["prompt_id"],
                               "reason": "no matching baseline samples"})
            continue
        mismatch_arms = []
        for baseline in references:
            same = (row["response"].get("content") == baseline["response"].get("content") and
                    row["response"].get("tokens") == baseline["response"].get("tokens"))
            if not same:
                mismatch_arms.append(baseline["arm"])
                mismatches.append({"arm": row["arm"], "mode": row["mode"],
                                   "repeat": row["repeat"], "prompt_id": row["prompt_id"],
                                   "baseline_arm": baseline["arm"]})
        comparisons.append({"arm": row["arm"], "mode": row["mode"],
                            "repeat": row["repeat"], "prompt_id": row["prompt_id"],
                            "baseline_arms_compared": [baseline["arm"] for baseline in references],
                            "equivalent_to_all_baselines": not mismatch_arms})
    return {
        "compared_against_all_baseline_samples": True,
        "baseline_sample_count": sum(len(rows) for rows in baselines.values()),
        "comparison_count": sum(len(baselines.get((row["repeat"], row["prompt_id"]), []))
                                for row in records),
        "comparisons": comparisons,
        "mismatches": mismatches,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprints(server_path, runtime_environment):
    """Record executable and resolved libllama hashes when they can be read."""
    server_path = Path(server_path).resolve()
    result = {
        "server_executable": {"path": str(server_path), "sha256": None},
        "libllama": {"path": None, "sha256": None, "status": "not-resolved"},
    }
    try:
        result["server_executable"]["sha256"] = _sha256_file(server_path)
    except OSError as error:
        result["server_executable"]["error"] = type(error).__name__
    try:
        ldd = subprocess.run(["ldd", str(server_path)], capture_output=True, text=True,
                             timeout=10, check=False, env=runtime_environment)
    except (OSError, subprocess.TimeoutExpired) as error:
        result["libllama"]["status"] = f"ldd-unavailable:{type(error).__name__}"
        return result
    for line in ldd.stdout.splitlines():
        if not re.match(r"\s*libllama\.so(?:\S*)\s+=>", line):
            continue
        resolved_path = line.split("=>", 1)[1].split("(", 1)[0].strip()
        if not resolved_path or resolved_path == "not found":
            result["libllama"]["status"] = "not-found"
            return result
        result["libllama"]["path"] = str(Path(resolved_path).resolve())
        try:
            result["libllama"]["sha256"] = _sha256_file(resolved_path)
            result["libllama"]["status"] = "resolved"
        except OSError as error:
            result["libllama"]["status"] = f"hash-unavailable:{type(error).__name__}"
        return result
    result["libllama"]["status"] = "not-listed-by-ldd" if ldd.returncode == 0 else "ldd-failed"
    return result


def _parse_pressure(path):
    result = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        values = {}
        for item in parts[1:]:
            key, value = item.split("=", 1)
            if key.startswith("avg"):
                parsed = float(value)
                values[key] = parsed if math.isfinite(parsed) else value
            else:
                values[key] = int(value)
        result[parts[0]] = values
    return result


def _nvidia_value(value):
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    return int(number) if number.is_integer() else number


def _cgroup_v2_path(pid):
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            relative = Path(line[3:].lstrip("/"))
            if ".." in relative.parts:
                raise ValueError("invalid unified cgroup path")
            return Path("/sys/fs/cgroup") / relative
    return None


def _integer_key_values(text):
    return {key: int(value) for key, value in (line.split() for line in text.splitlines())}


class ResourceSampler:
    """One low-rate thread captures process, host and NVIDIA metrics per server arm."""

    def __init__(self, pid, interval=1.0):
        self.pid = pid
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="bench-resource-sampler", daemon=True)
        self.gpu_disabled = False
        self.cgroup_path = None
        try:
            self.cgroup_path = _cgroup_v2_path(pid)
        except (OSError, ValueError) as error:
            self._record_error("target_process_cgroup", error)

    def start(self):
        self.thread.start()

    def _record_error(self, source, error):
        self.errors.append({"timestamp": time.time(), "source": source,
                            "error": f"{type(error).__name__}: {error}"})

    def _capture(self):
        sample = {"timestamp": time.time(), "monotonic_s": time.monotonic()}
        try:
            meminfo = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    meminfo["MemAvailable_kB"] = int(line.split()[1])
                    break
            sample["os_memory"] = meminfo
        except Exception as error:  # Sampling is best-effort and cannot invalidate a request.
            sample["os_memory"] = None
            self._record_error("/proc/meminfo", error)
        sample["pressure"] = {}
        for kind in ("memory", "cpu", "io"):
            try:
                sample["pressure"][kind] = _parse_pressure(Path(f"/proc/pressure/{kind}"))
            except Exception as error:
                self._record_error(f"/proc/pressure/{kind}", error)
        try:
            status = Path(f"/proc/{self.pid}/status").read_text().splitlines()
            process_memory = {}
            for line in status:
                key, separator, value = line.partition(":")
                if separator and key in ("VmRSS", "VmSwap", "RssAnon", "RssFile"):
                    fields = value.split()
                    process_memory[f"{key}_kB"] = int(fields[0]) if fields else None
            sample["process_memory"] = process_memory
        except Exception as error:
            sample["process_memory"] = None
            self._record_error("target_process_status", error)
        if self.cgroup_path is not None:
            try:
                sample["cgroup_memory"] = {
                    "path": str(self.cgroup_path),
                    "current_bytes": int((self.cgroup_path / "memory.current").read_text()),
                    "limit_bytes": _nvidia_value((self.cgroup_path / "memory.max").read_text()),
                    "swap_current_bytes": int((self.cgroup_path / "memory.swap.current").read_text()),
                    "events": _integer_key_values((self.cgroup_path / "memory.events").read_text()),
                    "stat": _integer_key_values((self.cgroup_path / "memory.stat").read_text()),
                }
            except Exception as error:
                sample["cgroup_memory"] = None
                self._record_error("target_cgroup_memory", error)
        if not self.gpu_disabled:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,clocks.sm,clocks.mem,power.draw,pstate",
                     "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=2,
                    env={"PATH": os.environ.get("PATH", os.defpath)},
                )
                sample["gpu"] = []
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row) == 8:
                        sample["gpu"].append({
                            "utilization_pct": _nvidia_value(row[0]),
                            "temperature_c": _nvidia_value(row[1]),
                            "memory_used_mib": _nvidia_value(row[2]),
                            "memory_total_mib": _nvidia_value(row[3]),
                            "sm_clock_mhz": _nvidia_value(row[4]),
                            "memory_clock_mhz": _nvidia_value(row[5]),
                            "power_draw_w": _nvidia_value(row[6]),
                            "pstate": _nvidia_value(row[7]),
                        })
            except FileNotFoundError as error:
                sample["gpu"] = None
                self._record_error("nvidia-smi", error)
                self.gpu_disabled = True
            except Exception as error:
                sample["gpu"] = None
                self._record_error("nvidia-smi", error)
        self.samples.append(sample)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self._capture()
            except Exception as error:
                self._record_error("sampler", error)
            self.stop_event.wait(self.interval)

    def stop_and_save(self, path):
        self.stop_event.set()
        if self.thread.ident is None:
            self._record_error("sampler", RuntimeError("sampling thread did not start"))
        else:
            self.thread.join()
        write_json(path, {
            "pid": self.pid,
            "interval_s": self.interval,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "sampling_errors": self.errors,
        })


def assert_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # A terminated arm can leave client connections in TIME_WAIT. Match
        # the server's reuse policy without permitting a second live listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"port {port} is already bound; refusing to benchmark an unknown server") from error


def stop_server(server):
    if server.poll() is not None:
        return
    try:
        os.killpg(server.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait()


def main(argv=None):
    # A surrounding timeout must still execute the server's cleanup block.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = validate_args(args, parser)
    args.output.mkdir(parents=True, exist_ok=False)
    prompts = [
        "Write a detailed numbered explanation of how a GPU executes a neural network. "
        "Cover computation, memory, caching, and parallel execution. Continue for at least 300 words.",
        "Background notes:\n" + "The workload uses sparse mixture-of-experts routing, "
        "limited GPU memory, a host cache, and on-demand storage reads.\n" * 32 +
        "Explain in detail how to measure throughput and find bottlenecks without fabricating numbers. "
        "Continue for at least 300 words.",
    ]
    runtime_environment = {key: os.environ[key] for key in RUNTIME_ENV_KEYS if key in os.environ}
    fingerprints = artifact_fingerprints(args.server, runtime_environment)
    run_metadata = {
        "mode_order": modes,
        "started_unix_s": time.time(),
        "artifacts": fingerprints,
        "parameters": {
            "tokens": args.tokens, "repeats": args.repeats, "port": args.port,
            "gpu_cache_mib": args.gpu_cache_mib, "pinned_cache_mib": args.pinned_cache_mib,
            "speculative_slots": args.speculative_slots, "context": args.context,
            "batch": args.batch, "ubatch": args.ubatch, "threads": args.threads,
            "profile": args.profile, "reserve_always": args.reserve_always,
        },
        "runtime_environment": runtime_environment,
    }
    write_json(args.output / "run.json", run_metadata)
    records = []
    for index, mode in enumerate(modes):
        # Inherit only runtime variables needed to locate libraries/devices and temp files.
        env = dict(runtime_environment)
        env.update(LLAMA_MOE_EXPERT_USED_OVERRIDE="4", GGML_CUDA_DISABLE_GRAPHS="1",
                   LLAMA_DSTORAGE_SUMMARY="1", LLAMA_DSTORAGE_PREFILL_WORKSPACE="0",
                   LLAMA_DSTORAGE_EVICT_TO_RAM="1",
                   LLAMA_DSTORAGE_SPECULATIVE_SLOTS=str(args.speculative_slots),
                   LLAMA_GDS_READ_THREADS="20")
        env["LLAMA_DSTORAGE_SPECULATIVE_RESERVE_ALWAYS"] = "1" if args.reserve_always else "0"
        if args.profile:
            env["LLAMA_DSTORAGE_PREROUTER_PROFILE"] = "1"
        if mode != "baseline":
            env["LLAMA_DSTORAGE_PREROUTER_MODE"] = mode
        command = [str(args.server.resolve()), "-m", str(args.model.resolve()), "-ngl", "99",
                   "-c", str(args.context), "-b", str(args.batch), "-ub", str(args.ubatch),
                   "-t", str(args.threads), "-tb", str(args.threads),
                   "-ctk", "tq3_0", "-ctv", "tq3_0", "--flash-attn", "on",
                   "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                   "--spec-draft-p-min", "0", "--cache-ram", "0", "--ctx-checkpoints", "0",
                   "--reasoning", "off", "--no-warmup", "--fit", "off", "--jinja",
                   "--no-cont-batching", "--parallel", "1", "--metrics", "--host", "127.0.0.1",
                   "--port", str(args.port), "--dstorage-moe", "--dstorage-moe-prefetch",
                   "--moe-gpu-cache-mib", str(args.gpu_cache_mib),
                   "--moe-pinned-cache-mib", str(args.pinned_cache_mib)]
        if mode != "baseline":
            command += ["--dstorage-prerouter", str(args.checkpoint.resolve())]
        stem = f"{index:02d}-{mode}"
        metadata = {
            "command": command, "mode": mode, "arm": index,
            "environment": {key: env[key] for key in CONTROLLED_ENV_KEYS if key in env},
            "runtime_environment_keys": sorted(runtime_environment),
        }
        write_json(args.output / f"{stem}-config.json", metadata)
        assert_port_available(args.port)
        with (args.output / f"{stem}-server.log").open("w") as logfile:
            server = subprocess.Popen(command, env=env, stdout=logfile, stderr=logfile,
                                      start_new_session=True)
            sampler = ResourceSampler(server.pid)
            try:
                sampler.start()
                deadline = time.monotonic() + 120
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"{mode} server exited: {server.returncode}; see {stem}-server.log")
                    try:
                        if request(args.port, "/health", timeout=2).get("status") == "ok":
                            break
                    except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("server health timeout")
                    time.sleep(0.25)
                for repeat in range(args.repeats):
                    for prompt_id, prompt in enumerate(prompts):
                        start = time.monotonic()
                        response = request(args.port, "/completion", {
                            "prompt": prompt, "n_predict": args.tokens, "temperature": 0,
                            "seed": 1, "cache_prompt": False, "stream": False,
                            "ignore_eos": True, "return_tokens": True,
                        })
                        try:
                            validate_completion(response, args.tokens)
                        except ValueError as error:
                            raise RuntimeError(f"invalid or incomplete completion in {stem}: {error}; "
                                               f"response={response}") from error
                        record = {"arm": index, "mode": mode, "repeat": repeat,
                                  "prompt_id": prompt_id, "wall_s": time.monotonic() - start,
                                  "response": response}
                        records.append(record)
                        write_json(args.output / "results.json", records)
                        label = f"{stem}-{repeat}-{prompt_id}"
                        (args.output / f"{label}-metrics.txt").write_text(request(args.port, "/metrics"))
                        print(label, json.dumps(response.get("timings", {})), flush=True)
            finally:
                try:
                    sampler.stop_and_save(args.output / f"{stem}-resources.json")
                finally:
                    stop_server(server)

    equivalence = equivalence_report(records)
    write_json(args.output / "equivalence.json", equivalence)
    summary = {
        "mode_order": modes,
        "mode_summaries": {mode: summarize_records(records, mode=mode)
                           for mode in dict.fromkeys(modes)},
        "prompt_summaries": {str(prompt_id): summarize_records(records, prompt_id=prompt_id)
                             for prompt_id in range(len(prompts))},
        "overall": summarize_records(records),
        "equivalence_mismatch_count": len(equivalence["mismatches"]),
    }
    write_json(args.output / "summary.json", summary)
    for mode in dict.fromkeys(modes):
        write_json(args.output / "summary" / "modes" / mode / "summary.json",
                   summary["mode_summaries"][mode])
    for prompt_id, prompt_summary in summary["prompt_summaries"].items():
        write_json(args.output / "summary" / "prompts" / f"{int(prompt_id):02d}" / "summary.json",
                   prompt_summary)
    mismatch_count = len(equivalence["mismatches"])
    print(f"Completed; output/token mismatches: {mismatch_count}; evidence: {args.output}", flush=True)
    return 1 if mismatch_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
