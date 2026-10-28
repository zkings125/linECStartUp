"""Machine-readable P0 physics readiness audit.

The audit verifies evidence files rather than trusting boolean switches.  A
verified file is still distinguished from a solver path that consumes it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

from .config import SimulationConfig
from .validation_manifest import audit_frozen_validation_manifest, sha256_file


@dataclass(frozen=True)
class P0Gate:
    name: str
    status: str
    implemented: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["implemented"] = list(self.implemented)
        payload["blockers"] = list(self.blockers)
        return payload


@dataclass(frozen=True)
class P0Readiness:
    device: str
    shot_id: str | None
    quantitative_ready: bool
    gates: tuple[P0Gate, ...]
    evidence_manifest: str | None

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "shot_id": self.shot_id,
            "quantitative_ready": self.quantitative_ready,
            "gates": [item.to_dict() for item in self.gates],
            "evidence_manifest": self.evidence_manifest,
            "claim_policy": (
                "quantitative prediction is blocked unless every P0 gate is complete"
            ),
        }


def _load_evidence(path: str | Path | None) -> tuple[dict[str, Mapping[str, object]], Path | None, list[str]]:
    if path is None:
        return {}, None, ["P0 evidence manifest was not supplied"]
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    errors: list[str] = []
    if str(payload.get("schema_version")) != "1.0":
        errors.append("P0 evidence schema_version must be '1.0'")
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        errors.append("P0 evidence artifacts must be an object")
        return {}, source, errors
    verified: dict[str, Mapping[str, object]] = {}
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            errors.append(f"artifact {name} must be an object")
            continue
        artifact = (source.parent / str(record.get("path", ""))).resolve()
        expected = str(record.get("sha256", "")).lower()
        if not artifact.is_file():
            errors.append(f"artifact {name} is missing: {artifact}")
        elif len(expected) != 64 or sha256_file(artifact) != expected:
            errors.append(f"artifact {name} SHA-256 mismatch")
        elif str(record.get("status", "")) != "verified":
            errors.append(f"artifact {name} status must be 'verified'")
        else:
            verified[str(name)] = record
    return verified, source, errors


def assess_p0_readiness(
    config: SimulationConfig,
    evidence_manifest: str | Path | None = None,
) -> P0Readiness:
    artifacts, evidence_path, evidence_errors = _load_evidence(evidence_manifest)

    def missing(required: set[str]) -> list[str]:
        return [f"verified artifact: {name}" for name in sorted(required - set(artifacts))]

    field_blockers = missing({
        "shot_coil_waveforms", "mutual_inductance", "eddy_structure",
        "first_wall", "orbit_loss_validation",
    }) + evidence_errors
    gate1 = P0Gate(
        "shot_fields_wall_orbit",
        "complete" if not field_blockers else "blocked",
        (
            "PF/TF/CS/TPC/VFC Biot-Savart providers",
            "mutual-inductance network",
            "polygonal wall field-line hit and topology scan",
        ),
        tuple(field_blockers),
    )

    chemistry_blockers = missing({
        "molecular_reaction_database", "material_recycling_database", "material_sputtering_database"
    })
    # The present normalized cross-section adapter is real, but the molecular
    # reaction-state network is not yet wired into species_wall.
    chemistry_blockers.append("H2/D2 vibrational/molecular-ion state network is not coupled to the global species ODE")
    gate2 = P0Gate(
        "molecular_wall_database",
        "complete" if not chemistry_blockers else "blocked",
        ("checksummed normalized cross-section adapter", "OpenADAS atomic H/D charge-state backend"),
        tuple(chemistry_blockers),
    )

    kinetic_blockers: list[str] = []
    if config.kinetic.backend != "open_field_line_1d2v":
        kinetic_blockers.append("kinetic.backend must be 'open_field_line_1d2v'")
    if config.kinetic.coupling_mode != "time_dependent_feedback":
        kinetic_blockers.append("kinetic.coupling_mode must be 'time_dependent_feedback'")
    if not config.kinetic.spatial_fieldline_file:
        kinetic_blockers.append("spatial_fieldline_file is required")
    if not config.kinetic.spatial_dql_file:
        kinetic_blockers.append("spatial_dql_file is required for powered coupling")
    gate3 = P0Gate(
        "main_workflow_1d2v",
        "complete" if not kinetic_blockers else "blocked",
        (
            "time-dependent fixed-point current feedback",
            "replacement ionization/absorption/wall-loss closure",
            "implicit open-line streaming and optional ambipolar sheath",
        ),
        tuple(kinetic_blockers),
    )

    gate4 = P0Gate(
        "direct_xb_accessibility",
        "complete",
        ("R/L cutoff", "UHR", "evanescent X layer", "fast/slow-X branch labels"),
        (),
    )

    wave_blockers = missing({"nonlocal_hot_dielectric", "mode_decomposition", "fullwave_power_momentum_closure"})
    wave_blockers.append("internal Nedelec solver uses a local hot-loss tensor, not a nonlocal hot dielectric")
    gate5 = P0Gate(
        "nonlocal_hot_wave_fp_closure",
        "complete" if not wave_blockers else "blocked",
        (
            "verified external full-wave NPZ import",
            "O/X/EBW identity metadata gate",
            "power-normalized spatial non-diagonal Dql projection",
        ),
        tuple(wave_blockers),
    )

    validation_blockers = missing({"frozen_validation_manifest"})
    record = artifacts.get("frozen_validation_manifest")
    if record is not None and evidence_path is not None:
        validation_path = (evidence_path.parent / str(record["path"])).resolve()
        validation_audit = audit_frozen_validation_manifest(validation_path)
        validation_blockers.extend(validation_audit.errors)
    gate6 = P0Gate(
        "frozen_fit_blind_validation",
        "complete" if not validation_blockers else "blocked",
        ("checksummed non-overlap audit", "Ip/ne/Balmer-alpha/HXR-or-ECE/loop-voltage channel gate"),
        tuple(validation_blockers),
    )
    gates = (gate1, gate2, gate3, gate4, gate5, gate6)
    return P0Readiness(
        config.validation.device_name,
        config.validation.shot_id,
        all(item.status == "complete" for item in gates),
        gates,
        None if evidence_path is None else str(evidence_path),
    )


__all__ = ["P0Gate", "P0Readiness", "assess_p0_readiness"]
