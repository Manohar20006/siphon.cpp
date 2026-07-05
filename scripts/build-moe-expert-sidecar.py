#!/usr/bin/env python3
"""Build a 4K-aligned sidecar file for GGUF MoE expert tensors.

The runtime consumes the emitted manifest via:

    LLAMA_MOE_EXPERT_SIDECAR_MANIFEST=/path/to/manifest.tsv

Tensor-major manifest format:

    layer<TAB>tensor_name<TAB>sidecar_path<TAB>sidecar_offset<TAB>total_size<TAB>n_experts

Expert-major manifest format appends:

    expert_major<TAB>record_stride<TAB>tensor_offset
"""

from __future__ import annotations

import argparse
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
EXPERT_RE = re.compile(
    rb"^blk\.(\d+)\.ffn_(?:gate_up_exps|down_exps|gate_exps|up_exps)\.weight$"
)


# ggml_type -> (block_size, type_size)
# Covers the standard GGML types used by modern GGUF quantized models.
GGML_TYPE_INFO = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 40),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    30: (1, 2),      # BF16
}


@dataclass
class TensorInfo:
    name: str
    layer: int
    dims: list[int]
    ggml_type: int
    rel_offset: int
    size: int
    n_experts: int


def read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"unexpected EOF while reading {n} bytes")
    return data


def read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_i64(f: BinaryIO) -> int:
    return struct.unpack("<q", read_exact(f, 8))[0]


def read_f64(f: BinaryIO) -> float:
    return struct.unpack("<d", read_exact(f, 8))[0]


def read_string(f: BinaryIO) -> bytes:
    n = read_u64(f)
    return read_exact(f, n)


def skip_value(f: BinaryIO, value_type: int) -> bytes | int | None:
    if value_type in (0, 1, 7):  # uint8, int8, bool
        read_exact(f, 1)
    elif value_type in (2, 3):  # uint16, int16
        read_exact(f, 2)
    elif value_type in (4, 5, 6):  # uint32, int32, float32
        data = read_exact(f, 4)
        if value_type == 4:
            return struct.unpack("<I", data)[0]
    elif value_type == 8:  # string
        return read_string(f)
    elif value_type in (10, 11, 12):  # uint64, int64, float64
        if value_type == 10:
            return read_u64(f)
        if value_type == 11:
            return read_i64(f)
        return read_f64(f)
    elif value_type == 9:  # array
        elem_type = read_u32(f)
        n = read_u64(f)
        for _ in range(n):
            skip_value(f, elem_type)
    else:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")
    return None


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_nbytes(dims: list[int], ggml_type: int) -> int:
    if ggml_type not in GGML_TYPE_INFO:
        raise ValueError(f"unsupported GGML tensor type {ggml_type}")
    block_size, type_size = GGML_TYPE_INFO[ggml_type]
    if not dims:
        raise ValueError("tensor has no dimensions")
    if dims[0] % block_size != 0:
        raise ValueError(f"tensor first dimension {dims[0]} is not divisible by block size {block_size}")
    nbytes = type_size * (dims[0] // block_size)
    for dim in dims[1:]:
        nbytes *= dim
    return nbytes


def parse_gguf(path: Path) -> tuple[int, list[TensorInfo]]:
    with path.open("rb") as f:
        if read_exact(f, 4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = read_u32(f)
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = read_u64(f)
        n_kv = read_u64(f)

        alignment = DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            key = read_string(f)
            value_type = read_u32(f)
            value = skip_value(f, value_type)
            if key == b"general.alignment" and isinstance(value, int):
                alignment = int(value)

        tensors: list[TensorInfo] = []
        for _ in range(n_tensors):
            name_raw = read_string(f)
            n_dims = read_u32(f)
            dims = [read_u64(f) for _ in range(n_dims)]
            ggml_type = read_u32(f)
            rel_offset = read_u64(f)

            match = EXPERT_RE.match(name_raw)
            if not match:
                continue
            name = name_raw.decode("utf-8")
            layer = int(match.group(1))
            size = tensor_nbytes(dims, ggml_type)
            n_experts = int(dims[2]) if len(dims) >= 3 else 0
            if n_experts <= 0 or size % n_experts != 0:
                raise ValueError(f"expert tensor {name} has invalid expert dimension/size")
            tensors.append(TensorInfo(name, layer, dims, ggml_type, rel_offset, size, n_experts))

        data_offset = align_up(f.tell(), alignment)
        return data_offset, tensors


def copy_range(src: BinaryIO, dst: BinaryIO, src_offset: int, size: int, chunk_size: int) -> None:
    src.seek(src_offset)
    remaining = size
    while remaining > 0:
        chunk = src.read(min(chunk_size, remaining))
        if not chunk:
            raise EOFError(f"unexpected EOF copying range at offset {src_offset}")
        dst.write(chunk)
        remaining -= len(chunk)


def build_tensor_major_sidecar(
    tensors: list[TensorInfo],
    data_offset: int,
    model: Path,
    sidecar: Path,
    manifest: Path,
    alignment: int,
    chunk_size: int,
    dry_run: bool,
) -> int:
    sidecar_abs = sidecar.resolve()
    out_offset = 0
    manifest_lines: list[str] = [
        "# layer\ttensor_name\tsidecar_path\tsidecar_offset\ttotal_size\tn_experts\tlayout\trecord_stride\ttensor_offset\n"
    ]

    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    manifest_tmp = manifest.with_name(manifest.name + ".tmp")

    src: BinaryIO | None = None
    dst: BinaryIO | None = None
    try:
        if not dry_run:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if sidecar_tmp.exists():
                sidecar_tmp.unlink()
            if manifest_tmp.exists():
                manifest_tmp.unlink()
            src = model.open("rb")
            dst = sidecar_tmp.open("wb")

        for tensor in sorted(tensors, key=lambda t: (t.layer, t.name)):
            out_offset = align_up(out_offset, alignment)
            src_offset = data_offset + tensor.rel_offset
            expert_stride = tensor.size // tensor.n_experts
            if expert_stride % 4096 != 0:
                print(f"warning: expert stride for {tensor.name} is not 4K aligned: {expert_stride}")

            print(
                f"L{tensor.layer:02d} {tensor.name} src={src_offset} dst={out_offset} "
                f"size={tensor.size / 1024 / 1024:.2f} MiB experts={tensor.n_experts}"
            )

            manifest_lines.append(
                f"{tensor.layer}\t{tensor.name}\t{sidecar_abs}\t{out_offset}\t{tensor.size}\t{tensor.n_experts}"
                f"\ttensor_major\t0\t0\n"
            )

            if not dry_run:
                assert src is not None and dst is not None
                pad = out_offset - dst.tell()
                if pad > 0:
                    dst.write(b"\0" * pad)
                copy_range(src, dst, src_offset, tensor.size, chunk_size)
            out_offset += tensor.size

        if not dry_run:
            manifest_tmp.write_text("".join(manifest_lines), encoding="utf-8")
            assert dst is not None
            dst.flush()
            os.fsync(dst.fileno())
            dst.close()
            dst = None
            os.replace(sidecar_tmp, sidecar)
            os.replace(manifest_tmp, manifest)
    finally:
        if src is not None:
            src.close()
        if dst is not None:
            dst.close()
        if not dry_run:
            if sidecar_tmp.exists():
                print(f"partial sidecar remains at: {sidecar_tmp}")
            if manifest_tmp.exists():
                print(f"partial manifest remains at: {manifest_tmp}")

    return out_offset


def build_expert_major_sidecar(
    tensors: list[TensorInfo],
    data_offset: int,
    model: Path,
    sidecar: Path,
    manifest: Path,
    alignment: int,
    chunk_size: int,
    dry_run: bool,
) -> int:
    sidecar_abs = sidecar.resolve()
    out_offset = 0
    manifest_lines: list[str] = [
        "# layer\ttensor_name\tsidecar_path\tsidecar_offset\ttotal_size\tn_experts\tlayout\trecord_stride\ttensor_offset\n"
    ]

    tensors_by_layer: dict[int, list[TensorInfo]] = {}
    for tensor in tensors:
        tensors_by_layer.setdefault(tensor.layer, []).append(tensor)

    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    manifest_tmp = manifest.with_name(manifest.name + ".tmp")

    src: BinaryIO | None = None
    dst: BinaryIO | None = None
    try:
        if not dry_run:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if sidecar_tmp.exists():
                sidecar_tmp.unlink()
            if manifest_tmp.exists():
                manifest_tmp.unlink()
            src = model.open("rb")
            dst = sidecar_tmp.open("wb")

        for layer in sorted(tensors_by_layer):
            layer_tensors = sorted(tensors_by_layer[layer], key=lambda t: t.name)
            n_experts = layer_tensors[0].n_experts
            if any(t.n_experts != n_experts for t in layer_tensors):
                raise ValueError(f"layer {layer} has inconsistent expert counts")

            layer_base = align_up(out_offset, alignment)
            tensor_offsets: dict[str, int] = {}
            record_stride = 0
            for tensor in layer_tensors:
                record_stride = align_up(record_stride, alignment)
                tensor_offsets[tensor.name] = record_stride
                expert_stride = tensor.size // tensor.n_experts
                if expert_stride % alignment != 0:
                    print(f"warning: expert stride for {tensor.name} is not {alignment}-aligned: {expert_stride}")
                record_stride += expert_stride
            record_stride = align_up(record_stride, alignment)

            print(
                f"L{layer:02d} expert_major base={layer_base} record_stride={record_stride} "
                f"experts={n_experts} tensors={len(layer_tensors)}"
            )

            for tensor in layer_tensors:
                manifest_lines.append(
                    f"{tensor.layer}\t{tensor.name}\t{sidecar_abs}\t{layer_base}\t{tensor.size}\t{tensor.n_experts}"
                    f"\texpert_major\t{record_stride}\t{tensor_offsets[tensor.name]}\n"
                )

            if not dry_run:
                assert src is not None and dst is not None
                pad = layer_base - dst.tell()
                if pad > 0:
                    dst.write(b"\0" * pad)
                for eid in range(n_experts):
                    record_base = layer_base + eid * record_stride
                    for tensor in layer_tensors:
                        expert_stride = tensor.size // tensor.n_experts
                        dst_offset = record_base + tensor_offsets[tensor.name]
                        pad = dst_offset - dst.tell()
                        if pad > 0:
                            dst.write(b"\0" * pad)
                        src_offset = data_offset + tensor.rel_offset + eid * expert_stride
                        copy_range(src, dst, src_offset, expert_stride, chunk_size)

            out_offset = layer_base + n_experts * record_stride

        if not dry_run:
            manifest_tmp.write_text("".join(manifest_lines), encoding="utf-8")
            assert dst is not None
            dst.flush()
            os.fsync(dst.fileno())
            dst.close()
            dst = None
            os.replace(sidecar_tmp, sidecar)
            os.replace(manifest_tmp, manifest)
    finally:
        if src is not None:
            src.close()
        if dst is not None:
            dst.close()
        if not dry_run:
            if sidecar_tmp.exists():
                print(f"partial sidecar remains at: {sidecar_tmp}")
            if manifest_tmp.exists():
                print(f"partial manifest remains at: {manifest_tmp}")

    return out_offset


def build_sidecar(
    model: Path,
    sidecar: Path,
    manifest: Path,
    alignment: int,
    chunk_size: int,
    dry_run: bool,
    layout: str,
) -> None:
    data_offset, tensors = parse_gguf(model)
    if not tensors:
        raise RuntimeError("no Qwen MoE expert tensors found")

    sidecar_abs = sidecar.resolve()
    manifest_abs = manifest.resolve()
    print(f"model: {model}")
    print(f"data_offset: {data_offset}")
    print(f"expert_tensors: {len(tensors)}")
    print(f"sidecar: {sidecar_abs}")
    print(f"manifest: {manifest_abs}")
    print(f"layout: {layout}")

    if layout == "tensor-major":
        out_offset = build_tensor_major_sidecar(
            tensors, data_offset, model, sidecar, manifest, alignment, chunk_size, dry_run)
    elif layout == "expert-major":
        out_offset = build_expert_major_sidecar(
            tensors, data_offset, model, sidecar, manifest, alignment, chunk_size, dry_run)
    else:
        raise ValueError(f"unsupported layout {layout}")

    print(f"sidecar_payload_end: {out_offset}")
    if dry_run:
        print("dry_run: did not write sidecar or manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--layout", choices=("tensor-major", "expert-major"), default="tensor-major")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.alignment <= 0 or args.alignment & (args.alignment - 1):
        raise ValueError("--alignment must be a power of two")
    build_sidecar(
        args.model,
        args.sidecar,
        args.manifest,
        args.alignment,
        args.chunk_mib * 1024 * 1024,
        args.dry_run,
        args.layout,
    )


if __name__ == "__main__":
    main()
