"""Immutable fit/validation split contracts for startup experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


REQUIRED_DIAGNOSTIC_GROUPS = {
    "Ip": ("Ip", "ip"),
    "ne": ("ne", "HCN_NE001"),
    "Balmer_alpha": ("Halpha", "H_alpha", "Dalpha", "D_alpha", "HA001"),
    "fast_electron": ("HXR", "ECE", "HXR001"),
    "loop_voltage": ("loop_voltage", "LOOPV"),
}
VALID_LABELS = {"success", "failure"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenShot:
    shot_id: str
    label: str
    data_file: str
    sha256: str
    channels: tuple[str, ...]


@dataclass(frozen=True)
class FrozenValidationAudit:
    device: str
    manifest_path: str
    manifest_sha256: str
    fit_shot_count: int
    validation_shot_count: int
    fit_labels: tuple[str, ...]
    validation_labels: tuple[str, ...]
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("fit_labels", "validation_labels", "errors", "warnings"):
            payload[key] = list(payload[key])
        return payload


def _read_shots(records: object, split: str, errors: list[str]) -> tuple[FrozenShot, ...]:
    if not isinstance(records, list):
        errors.append(f"{split}_shots must be a list")
        return ()
    shots: list[FrozenShot] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"{split}_shots[{index}] must be an object")
            continue
        try:
            shot = FrozenShot(
                shot_id=str(record["shot_id"]),
                label=str(record["label"]).lower(),
                data_file=str(record["data_file"]),
                sha256=str(record["sha256"]).lower(),
                channels=tuple(str(item) for item in record["channels"]),
            )
        except (KeyError, TypeError) as exc:
            errors.append(f"{split}_shots[{index}] missing/invalid field: {exc}")
            continue
        if shot.label not in VALID_LABELS:
            errors.append(f"shot {shot.shot_id}: label must be success or failure")
        if len(shot.sha256) != 64 or any(char not in "0123456789abcdef" for char in shot.sha256):
            errors.append(f"shot {shot.shot_id}: sha256 must be 64 lowercase hex characters")
        channel_set = set(shot.channels)
        for group, alternatives in REQUIRED_DIAGNOSTIC_GROUPS.items():
            if not channel_set.intersection(alternatives):
                errors.append(f"shot {shot.shot_id}: missing diagnostic group {group}")
        shots.append(shot)
    return tuple(shots)


def audit_frozen_validation_manifest(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> FrozenValidationAudit:
    """Validate a non-overlapping, checksummed fit/blind-validation split."""

    manifest = Path(path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    if str(payload.get("schema_version")) != "1.0":
        errors.append("schema_version must be '1.0'")
    if not payload.get("frozen_at"):
        errors.append("frozen_at is required")
    if not payload.get("label_authority"):
        errors.append("label_authority is required; heuristic labels are not sufficient")
    fit = _read_shots(payload.get("fit_shots"), "fit", errors)
    validation = _read_shots(payload.get("validation_shots"), "validation", errors)
    fit_ids = {item.shot_id for item in fit}
    validation_ids = {item.shot_id for item in validation}
    overlap = sorted(fit_ids.intersection(validation_ids))
    if overlap:
        errors.append(f"fit/validation shot overlap: {', '.join(overlap)}")
    if len(fit_ids) != len(fit):
        errors.append("fit_shots contains duplicate shot_id values")
    if len(validation_ids) != len(validation):
        errors.append("validation_shots contains duplicate shot_id values")
    for split, shots in (("fit", fit), ("validation", validation)):
        labels = {item.label for item in shots}
        missing = sorted(VALID_LABELS - labels)
        if missing:
            errors.append(f"{split} split lacks labels: {', '.join(missing)}")
    if verify_files:
        for shot in (*fit, *validation):
            source = (manifest.parent / shot.data_file).resolve()
            if not source.is_file():
                errors.append(f"shot {shot.shot_id}: data file not found: {source}")
                continue
            actual = sha256_file(source)
            if actual != shot.sha256:
                errors.append(f"shot {shot.shot_id}: data SHA-256 mismatch")
    else:
        warnings.append("data-file existence and SHA-256 were not verified")
    return FrozenValidationAudit(
        device=str(payload.get("device", "unknown")),
        manifest_path=str(manifest),
        manifest_sha256=sha256_file(manifest),
        fit_shot_count=len(fit),
        validation_shot_count=len(validation),
        fit_labels=tuple(sorted({item.label for item in fit})),
        validation_labels=tuple(sorted({item.label for item in validation})),
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


__all__ = [
    "FrozenShot",
    "FrozenValidationAudit",
    "REQUIRED_DIAGNOSTIC_GROUPS",
    "audit_frozen_validation_manifest",
    "sha256_file",
]
