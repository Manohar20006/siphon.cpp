"""Read and validate runnable routed-MoE GGUF files and standard split sets."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


_SPLIT_FILENAME = re.compile(r"^(?P<family>.+)-(?P<index>\d{5,})-of-(?P<count>\d{5,})$")
_SPLIT_KEYS = {"split.no", "split.count", "split.tensors.count"}
_AUXILIARY_TYPES = {"adapter", "imatrix", "mmproj", "projector"}
_AUXILIARY_NAMES = ("mmproj", "projector", "vision-encoder", "vision_encoder")
_PER_LAYER_TOKEN_EMBEDDING = "per_layer_token_embd.weight"
_GIB = 1024 ** 3
_MAX_SPLIT_COUNT = (1 << 16) - 1  # GGUF split.count is stored as UINT16.


def _split_filename(path: Path) -> tuple[str, int, int] | None:
    match = _SPLIT_FILENAME.fullmatch(path.stem)
    if match is None:
        return None
    return match.group("family"), int(match.group("index")), int(match.group("count"))


def _split_family_parts(folder: Path, family: str, count: int) -> dict[int, list[Path]]:
    named_parts: dict[int, list[Path]] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() != ".gguf":
            continue
        split = _split_filename(path)
        if split is None or split[0] != family or split[2] != count:
            continue
        named_parts.setdefault(split[1], []).append(path)
    return named_parts


def _ordered_named_parts(
    family: str,
    count: int,
    named_parts: dict[int, list[Path]],
) -> list[Path]:
    paths = []
    for index, candidates in sorted(named_parts.items()):
        standard_name = f"{family}-{index:05d}-of-{count:05d}.gguf"
        paths.extend(sorted(
            candidates,
            key=lambda path: (path.name != standard_name, path.name),
        ))
    return paths


def _split_ambiguity(named_parts: dict[int, list[Path]], count: int) -> str | None:
    duplicates = [
        (index, sorted(paths, key=lambda path: path.name))
        for index, paths in sorted(named_parts.items())
        if len(paths) > 1
    ]
    if not duplicates:
        return None
    details = "; ".join(
        f"part {index} of {count}: {', '.join(path.name for path in paths)}"
        for index, paths in duplicates
    )
    return f"Ambiguous split filenames; multiple files claim the same numeric index ({details})"


def model_part_paths(model_path: Path) -> list[Path]:
    """Return expected sibling paths from the conventional GGUF shard name."""
    path = Path(model_path).expanduser().resolve()
    split = _split_filename(path)
    if split is None:
        return [path]

    family, index, count = split
    if count < 1 or count > _MAX_SPLIT_COUNT or index < 1 or index > count:
        raise ValueError(f"Invalid standard split filename: {path.name}")
    return [
        path.with_name(f"{family}-{part:05d}-of-{count:05d}.gguf")
        for part in range(1, count + 1)
    ]


def model_fingerprint(parts: list[Path]) -> list[dict[str, Any]]:
    """Return ordered canonical path, size, and modification time for each part."""
    result = []
    for part in parts:
        path = Path(part).expanduser().resolve()
        stat = path.stat()
        result.append({
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return result


def _read_part(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OSError(f"GGUF part is missing or unreadable: {path}")
    try:
        from gguf.gguf_reader import GGUFReader

        reader = GGUFReader(path)
        fields = {
            key: field.contents()
            for key, field in reader.fields.items()
        }
        tensors = [(tensor.name, int(tensor.n_bytes)) for tensor in reader.tensors]
        return {"path": path, "fields": fields, "tensors": tensors}
    except OSError:
        raise
    except Exception as exc:
        raise ValueError(f"Unreadable GGUF file {path.name}: {exc}") from exc


def _integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"GGUF metadata {field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GGUF metadata {field} must be an integer") from exc


def _preview_integer(fields: dict[str, Any], key: str) -> int | None:
    try:
        return _integer(fields.get(key), key)
    except ValueError:
        return None


def _field_text(fields: dict[str, Any], key: str) -> str | None:
    value = fields.get(key)
    if value is None:
        return None
    return str(value)


def _base_metadata(
    *,
    name: str | None = None,
    architecture: str | None = None,
    part_count: int | None = None,
    parts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "architecture": architecture,
        "layers": None,
        "native_context": None,
        "expert_count": None,
        "expert_used_count": None,
        "artifact_gib": None,
        "dense_gib": None,
        "core_gib": None,
        "lookup_gib": None,
        "expert_gib": None,
        "tensor_gib": None,
        "part_count": part_count,
        "parts": parts or [],
    }


def _is_auxiliary(path: Path, fields: dict[str, Any]) -> bool:
    general_type = str(fields.get("general.type", "")).strip().lower()
    if general_type in _AUXILIARY_TYPES:
        return True
    filename = path.name.lower()
    return any(marker in filename for marker in _AUXILIARY_NAMES)


def _tensor_breakdown(parts: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate descriptor bytes into legacy dense, core, lookup, and expert sizes."""
    tensor_bytes = 0
    dense_bytes = 0
    core_bytes = 0
    lookup_bytes = 0
    expert_bytes = 0

    for part in parts:
        for tensor_name, byte_count in part["tensors"]:
            tensor_bytes += byte_count
            if "exps" in tensor_name.lower():
                expert_bytes += byte_count
                continue
            dense_bytes += byte_count
            if (
                tensor_name == _PER_LAYER_TOKEN_EMBEDDING
                or tensor_name.endswith("." + _PER_LAYER_TOKEN_EMBEDDING)
            ):
                lookup_bytes += byte_count
            else:
                core_bytes += byte_count

    return {
        "dense_gib": dense_bytes / _GIB,
        "core_gib": core_bytes / _GIB,
        "lookup_gib": lookup_bytes / _GIB,
        "expert_gib": expert_bytes / _GIB,
        "tensor_gib": tensor_bytes / _GIB,
    }


def _validate_metadata_and_tensors(
    parts: list[dict[str, Any]],
    paths: list[Path],
    filename_count: int | None,
) -> dict[str, Any]:
    if not parts:
        raise ValueError("The GGUF model set has no readable parts")

    first_fields = parts[0]["fields"]
    declared_architecture = _field_text(first_fields, "general.architecture")
    if not declared_architecture:
        raise ValueError("GGUF does not declare general.architecture")

    total_tensor_count = sum(len(part["tensors"]) for part in parts)
    declared_tensor_total: int | None = None
    declared_identity: dict[str, Any] = {}
    all_tensor_names: set[str] = set()
    tensor_sizes = _tensor_breakdown(parts)

    for index, part in enumerate(parts):
        fields = part["fields"]
        path = paths[index]
        actual_split_count = _integer(fields.get("split.count"), "split.count")
        actual_split_no = _integer(fields.get("split.no"), "split.no")
        actual_tensor_total = _integer(
            fields.get("split.tensors.count"), "split.tensors.count"
        )

        if filename_count is not None:
            if actual_split_count != filename_count:
                raise ValueError(
                    f"{path.name} declares split.count={actual_split_count}; "
                    f"its standard filename declares {filename_count} parts"
                )
            if actual_split_no != index:
                raise ValueError(
                    f"{path.name} declares split.no={actual_split_no}; "
                    f"expected split.no={index} for part {index + 1}"
                )
            if actual_tensor_total is None or actual_tensor_total <= 0:
                raise ValueError(
                    f"{path.name} does not declare a valid split.tensors.count"
                )
            if declared_tensor_total is None:
                declared_tensor_total = actual_tensor_total
            elif actual_tensor_total != declared_tensor_total:
                raise ValueError("Split parts disagree on split.tensors.count")
        else:
            if actual_split_count not in (None, 1):
                raise ValueError(
                    "Non-standard split filename: split GGUF files must use the standard "
                    "<family>-00001-of-000NN.gguf format; restore the original filenames"
                )
            if actual_split_no not in (None, 0):
                raise ValueError(f"Unsplit GGUF declares unexpected split.no={actual_split_no}")
            if actual_tensor_total is not None and actual_tensor_total != len(part["tensors"]):
                raise ValueError(
                    "split.tensors.count does not match the GGUF tensor descriptors"
                )

        for key, value in fields.items():
            if key in _SPLIT_KEYS or key.startswith("GGUF."):
                continue
            if key in declared_identity and declared_identity[key] != value:
                raise ValueError(f"Split parts disagree on declared metadata {key}")
            declared_identity.setdefault(key, value)

        for tensor_name, _ in part["tensors"]:
            if tensor_name in all_tensor_names:
                raise ValueError(f"duplicate tensor name across split parts: {tensor_name}")
            all_tensor_names.add(tensor_name)

    if filename_count is not None and declared_tensor_total != total_tensor_count:
        raise ValueError(
            f"Declared split tensor count {declared_tensor_total} does not match "
            f"the {total_tensor_count} tensor descriptors across the parts"
        )

    architecture = declared_architecture
    expert_count = _integer(
        first_fields.get(f"{architecture}.expert_count"), f"{architecture}.expert_count"
    )
    expert_used_count = _integer(
        first_fields.get(f"{architecture}.expert_used_count"),
        f"{architecture}.expert_used_count",
    )
    has_expert_tensors = tensor_sizes["expert_gib"] > 0

    if _is_auxiliary(paths[0], first_fields):
        raise ValueError("Auxiliary GGUF files cannot be imported as runnable models")

    general_type = first_fields.get("general.type")
    if general_type is not None and str(general_type).strip().lower() not in {"model", ""}:
        raise ValueError(f"Unsupported GGUF general.type: {general_type}")

    if not has_expert_tensors:
        if expert_count is not None or expert_used_count is not None:
            raise ValueError("GGUF declares expert metadata but contains no routed expert tensors")
        raise ValueError("Dense models are unsupported; Siphon Studio requires routed MoE tensors")
    if (
        expert_count is None
        or expert_used_count is None
        or expert_count <= 0
        or expert_used_count <= 0
    ):
        raise ValueError("Routed MoE GGUF must declare valid expert_count and expert_used_count")
    if expert_used_count > expert_count:
        raise ValueError("expert_used_count cannot exceed expert_count")

    first_name = _field_text(first_fields, "general.name")
    if not first_name:
        first_name = paths[0].stem
    artifact_gib = sum(path.stat().st_size for path in paths) / _GIB
    from .memory_budget import memory_layout
    return {
        "name": first_name,
        "architecture": architecture,
        "layers": _integer(
            first_fields.get(f"{architecture}.block_count"),
            f"{architecture}.block_count",
        ),
        "native_context": _integer(
            first_fields.get(f"{architecture}.context_length"),
            f"{architecture}.context_length",
        ),
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
        "artifact_gib": artifact_gib,
        **tensor_sizes,
        "memory_layout": memory_layout(first_fields, [tensor for part in parts for tensor in part["tensors"]]),
        "part_count": len(paths),
        "parts": [str(path.resolve()) for path in paths],
    }


def _inspect_paths(paths: list[Path], filename_count: int | None) -> dict[str, Any]:
    if not paths:
        raise ValueError("The GGUF model set has no expected parts")
    expected = [Path(path).expanduser().resolve() for path in paths]
    for index, path in enumerate(expected, start=1):
        if not path.is_file():
            raise ValueError(f"Missing part {index} of {len(expected)}: {path.name}")

    read_parts = [_read_part(path) for path in expected]
    return _validate_metadata_and_tensors(read_parts, expected, filename_count)


def inspect_model_files(model_path: Path) -> dict[str, Any]:
    """Validate a single GGUF or complete standard-named split MoE model."""
    path = Path(model_path).expanduser().resolve()
    if path.suffix.lower() != ".gguf":
        raise ValueError(f"Model path must point to a .gguf file: {path}")

    split = _split_filename(path)
    if split is None:
        part = _read_part(path)
        declared_count = _integer(part["fields"].get("split.count"), "split.count")
        if declared_count is not None and declared_count > 1:
            raise ValueError(
                "Non-standard split filename: split GGUF files must use the standard "
                "<family>-00001-of-000NN.gguf format; restore the original filenames"
            )
        return _inspect_paths([path], None)

    family, index, count = split
    if count < 1 or count > _MAX_SPLIT_COUNT or index < 1 or index > count:
        raise ValueError(f"Invalid standard split filename: {path.name}")
    paths = [
        path.with_name(f"{family}-{part:05d}-of-{count:05d}.gguf")
        for part in range(1, count + 1)
    ]
    ambiguity = _split_ambiguity(
        _split_family_parts(path.parent, family, count), count
    )
    if ambiguity:
        raise ValueError(ambiguity)
    return _inspect_paths(paths, count)


def _preview_metadata(
    paths: list[Path],
    *,
    name_fallback: str,
    part_count: int,
    preview_paths: list[Path] | None = None,
) -> dict[str, Any]:
    canonical_paths = [path.resolve() for path in paths]
    preview = _base_metadata(
        name=name_fallback,
        part_count=part_count,
        parts=[str(path) for path in canonical_paths],
    )
    readable: list[dict[str, Any]] = []
    for path in preview_paths if preview_paths is not None else canonical_paths:
        path = path.resolve()
        if path.is_file():
            try:
                readable.append(_read_part(path))
            except (OSError, ValueError):
                continue

    if not readable:
        return preview

    fields = readable[0]["fields"]
    architecture = _field_text(fields, "general.architecture")
    preview.update({
        "name": _field_text(fields, "general.name") or name_fallback,
        "architecture": architecture,
        "layers": _preview_integer(fields, f"{architecture}.block_count")
        if architecture else None,
        "native_context": _preview_integer(fields, f"{architecture}.context_length")
        if architecture else None,
        "expert_count": _preview_integer(fields, f"{architecture}.expert_count")
        if architecture else None,
        "expert_used_count": _preview_integer(fields, f"{architecture}.expert_used_count")
        if architecture else None,
    })
    if len(readable) == len(canonical_paths):
        try:
            preview["artifact_gib"] = sum(
                path.stat().st_size for path in canonical_paths
            ) / _GIB
        except OSError:
            return preview
        preview.update(_tensor_breakdown(readable))
    return preview


def _entry(
    paths: list[Path],
    *,
    fallback_name: str,
    part_count: int,
    metadata: dict[str, Any] | None = None,
    error: str = "",
    preview_paths: list[Path] | None = None,
) -> dict[str, Any]:
    canonical_paths = [path.resolve() for path in paths]
    first_path = canonical_paths[0]
    details = metadata or _preview_metadata(
        canonical_paths,
        name_fallback=fallback_name,
        part_count=part_count,
        preview_paths=preview_paths,
    )
    name = details.get("name") or fallback_name
    digest = hashlib.sha256(str(first_path).encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"model-{digest}",
        "name": name,
        "architecture": details.get("architecture"),
        "parts": [str(path) for path in canonical_paths],
        "path": str(first_path),
        "part_count": part_count,
        "artifact_gib": details.get("artifact_gib"),
        "available": not error,
        "error": error,
        "metadata": details,
    }


def _scan_split_group(
    folder: Path,
    family: str,
    count: int,
    named_parts: dict[int, list[Path]],
) -> dict[str, Any]:
    if count < 1 or count > _MAX_SPLIT_COUNT:
        actual_paths = [path.resolve() for path in _ordered_named_parts(family, count, named_parts)]
        fallback_name = family
        for path in actual_paths:
            try:
                name = _field_text(_read_part(path)["fields"], "general.name")
                if name:
                    fallback_name = name
                    break
            except (OSError, ValueError):
                continue
        return _entry(
            actual_paths,
            fallback_name=fallback_name,
            part_count=max(1, count),
            error=f"Invalid standard split filename total: {count}",
            preview_paths=actual_paths,
        )

    expected_paths = [
        folder / f"{family}-{index:05d}-of-{count:05d}.gguf"
        for index in range(1, count + 1)
    ]
    actual_paths = _ordered_named_parts(family, count, named_parts)
    fallback_name = family
    for path in actual_paths:
        try:
            raw_name = _field_text(_read_part(path)["fields"], "general.name")
            if raw_name:
                fallback_name = raw_name
                break
        except (OSError, ValueError):
            continue

    invalid_index = next((index for index in named_parts if index < 1 or index > count), None)
    if invalid_index is not None:
        reason = f"Filename part {invalid_index} is outside the declared family total of {count}"
        return _entry(
            expected_paths,
            fallback_name=fallback_name,
            part_count=count,
            error=reason,
            preview_paths=actual_paths,
        )

    ambiguity = _split_ambiguity(named_parts, count)
    if ambiguity:
        return _entry(
            actual_paths,
            fallback_name=fallback_name,
            part_count=count,
            error=ambiguity,
            preview_paths=actual_paths,
        )

    try:
        metadata = _inspect_paths(expected_paths, count)
        return _entry(
            expected_paths,
            fallback_name=fallback_name,
            part_count=count,
            metadata=metadata,
            preview_paths=actual_paths,
        )
    except (OSError, ValueError) as exc:
        return _entry(
            expected_paths,
            fallback_name=fallback_name,
            part_count=count,
            error=str(exc),
            preview_paths=actual_paths,
        )


def _scan_single(path: Path) -> dict[str, Any]:
    fallback_name = path.stem
    part = None
    try:
        part = _read_part(path)
        raw_name = _field_text(part["fields"], "general.name")
        if raw_name:
            fallback_name = raw_name
        declared_count = _integer(part["fields"].get("split.count"), "split.count")
        if declared_count is not None and declared_count > 1:
            raise ValueError(
                "Non-standard split filename: split GGUF files must use the standard "
                "<family>-00001-of-000NN.gguf format; restore the original filenames"
            )
        metadata = _inspect_paths([path], None)
        return _entry([path], fallback_name=fallback_name, part_count=1, metadata=metadata)
    except (OSError, ValueError) as exc:
        try:
            declared_count = (
                _integer(part["fields"].get("split.count"), "split.count")
                if part else None
            )
            part_count = max(1, declared_count or 1)
        except ValueError:
            part_count = 1
        return _entry([path], fallback_name=fallback_name, part_count=part_count, error=str(exc))


def scan_model_folder(folder: Path) -> list[dict[str, Any]]:
    """Scan only top-level GGUF files, preserving invalid sets as error entries."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise OSError(f"Model folder is not a readable directory: {root}")

    candidates = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".gguf"),
        key=lambda path: path.name,
    )
    split_groups: dict[tuple[str, int], dict[int, list[Path]]] = {}
    singles: list[Path] = []
    for path in candidates:
        split = _split_filename(path)
        if split is None:
            singles.append(path)
            continue
        family, index, count = split
        split_groups.setdefault((family, count), {}).setdefault(index, []).append(path)

    entries = [
        _scan_split_group(root, family, count, named_parts)
        for (family, count), named_parts in sorted(split_groups.items())
    ]
    entries.extend(_scan_single(path) for path in singles)
    return sorted(entries, key=lambda entry: (entry["path"], entry["name"]))
