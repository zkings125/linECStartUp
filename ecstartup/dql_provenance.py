"""Evidence and pre-CFS compatibility audit for quasilinear diffusion inputs.

``D_ql`` is defined by the wave spectrum and resonant particle dynamics; it
does not mathematically require closed magnetic surfaces.  Some external code
workflows nevertheless store it in flux coordinates.  This module keeps that
implementation dependency separate from the physical requirement and prevents
an O-X-B launch label from being treated as proof of EBW conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig


PRE_CFS_COORDINATES = {
    "laboratory_RZ_velocity",
    "laboratory_RZ_relativistic_momentum",
    "open_field_line_s_velocity",
    "open_field_line_s_relativistic_momentum",
}
CONFIRMED_CONVERSION = {"verified_fullwave", "experimentally_validated_reduced_model"}


@dataclass(frozen=True)
class DqlSourceAssessment:
    source_model: str
    availability: str
    generator: str | None
    coordinate_basis: str
    requires_closed_flux_surfaces: bool | None
    pre_cfs_compatible: bool
    power_normalized: bool
    wave_branch: str
    mode_conversion_status: str
    evidence_level: str
    quantitative_eligible: bool
    reason: str
    required_next_inputs: tuple[str, ...]
    source_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["required_next_inputs"] = list(self.required_next_inputs)
        return payload


def _metadata_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            return {}
        raw = str(np.asarray(archive["metadata_json"]).reshape(()).item())
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("D_ql metadata_json must decode to an object")
    return parsed


def _assess_external(model: str, path: Path) -> DqlSourceAssessment:
    metadata = _metadata_from_npz(path)
    coordinate = str(metadata.get("coordinate_basis", "unknown"))
    if metadata.get("schema") == "ecstartup.fullwave_spatial_dql.v1":
        coordinate_system = str(metadata.get("coordinate_system", "velocity"))
        coordinate = f"open_field_line_s_{coordinate_system}"
    pre_cfs = bool(metadata.get("pre_cfs_compatible", coordinate in PRE_CFS_COORDINATES))
    requires_flux = metadata.get("requires_closed_flux_surfaces")
    if requires_flux is not None:
        requires_flux = bool(requires_flux)
        pre_cfs = pre_cfs and not requires_flux
    power_normalized = bool(
        metadata.get("power_normalized", False)
        or metadata.get("target_absorbed_power_W") is not None
    )
    branch = str(metadata.get("wave_branch", "unknown"))
    conversion = str(metadata.get("conversion_evidence", "not_reported"))
    fullwave_schema = metadata.get("schema") == "ecstartup.fullwave_spatial_dql.v1"
    mesh_ok = bool(metadata.get("converged", False) and metadata.get("mesh_verified", False))
    budget = metadata.get("power_budget", {})
    balance = float(metadata.get("fullwave_power_balance_relative", float("inf")))
    local_balance = float(metadata.get("local_absorption_balance_relative", float("inf")))
    ebw_power = float(budget.get("ebw_out_W", 0.0)) if isinstance(budget, dict) else 0.0
    mode_decomposition = bool(metadata.get("mode_decomposition_verified", False))
    if fullwave_schema and ebw_power > 0.0 and mode_decomposition and branch in {"ebw", "mixed"}:
        conversion = "verified_fullwave"
    elif fullwave_schema and ebw_power > 0.0 and not mode_decomposition:
        branch = "unresolved_fullwave"
        conversion = "declared_ebw_power_without_verified_mode_decomposition"
    conversion_ok = branch != "ebw" or conversion in CONFIRMED_CONVERSION
    branch_resolved = branch in {"electromagnetic_ec", "ebw", "mixed"}
    quantitative = bool(
        pre_cfs
        and power_normalized
        and branch_resolved
        and conversion_ok
        and (
            not fullwave_schema
            or (mesh_ok and balance <= 1.0e-3 and local_balance <= 1.0e-3)
        )
        and bool(metadata.get("quantitative_eligible", fullwave_schema))
    )
    missing: list[str] = []
    if not pre_cfs:
        missing.append("laboratory-coordinate or open-field-line spatial mapping")
    if not power_normalized:
        missing.append("absorbed-power normalization")
    if branch == "ebw" and not conversion_ok:
        missing.append("verified O-X-B conversion evidence")
    if branch not in {"electromagnetic_ec", "ebw", "mixed"}:
        missing.append("wave-branch provenance (electromagnetic EC versus EBW)")
    if fullwave_schema and not mode_decomposition:
        missing.append("verified full-wave eigenmode or k-space branch decomposition")
    if fullwave_schema and not mesh_ok:
        missing.append("full-wave convergence and mesh verification")
    if not bool(metadata.get("quantitative_eligible", fullwave_schema)):
        missing.append("independent device validation before quantitative current use")
    return DqlSourceAssessment(
        source_model=model,
        availability="imported",
        generator=str(metadata.get("generator") or metadata.get("solver_name") or "external_unknown"),
        coordinate_basis=coordinate,
        requires_closed_flux_surfaces=requires_flux,
        pre_cfs_compatible=pre_cfs,
        power_normalized=power_normalized,
        wave_branch=branch,
        mode_conversion_status=conversion,
        evidence_level=("verified_external" if quantitative else "imported_unverified"),
        quantitative_eligible=quantitative,
        reason=(
            "external D_ql passes the declared pre-CFS provenance gates"
            if quantitative
            else "external arrays are readable but their metadata do not pass every pre-CFS quantitative gate"
        ),
        required_next_inputs=tuple(missing),
        source_file=str(path.resolve()),
    )


def assess_dql_source(
    config: SimulationConfig,
    *,
    source_file: str | Path | None = None,
) -> DqlSourceAssessment:
    """Assess the configured Dql without running a wave or kinetic solve."""

    model = config.kinetic.rf_diffusion_model
    path_value = source_file or config.kinetic.rf_diffusion_file
    if path_value is not None:
        return _assess_external(model, Path(path_value))
    if model == "zero":
        return DqlSourceAssessment(
            model, "unavailable", None, "none", False, True, False, "none",
            "not_evaluated", "none", False,
            "no quasilinear diffusion source is configured",
            ("internal pre-CFS wave calculation or a provenance-complete external D_ql",),
        )
    if model == "synthetic_gaussian":
        return DqlSourceAssessment(
            model, "generated", "linEC_synthetic_gaussian", "velocity_only", False,
            True, False, "surrogate", "not_evaluated", "sensitivity_only", False,
            "a Gaussian velocity-space shape has no spatial wave or mode-conversion evidence",
            ("laboratory-coordinate wave fields", "absorbed-power normalization"),
        )
    if model == "genray_like_internal":
        launch_modes = [item.launch.mode.upper() for item in config.rf.sources if item.enabled]
        requests_oxb = any("B" in mode or "O-X" in mode for mode in launch_modes)
        return DqlSourceAssessment(
            model, "generated", "linEC_reduced_genray_like",
            (
                "laboratory_RZ_relativistic_momentum"
                if config.kinetic.coordinate_system == "relativistic_momentum"
                else "laboratory_RZ_velocity"
            ),
            False, True, True,
            "electromagnetic_ec_or_unresolved_oxb" if requests_oxb else "electromagnetic_ec",
            "geometric_candidate_not_fullwave" if requests_oxb else "not_applicable",
            "internal_reduced_physics", False,
            (
                "internally generated from MagneticFieldProvider without GS/EFIT; an O-X-B label "
                "does not convert this cold-ray D_ql into confirmed EBW diffusion"
            ),
            (
                "verified hot full-wave O-X-B fields and signed EBW spectrum"
                if requests_oxb else
                "device validation of absorption, spectrum and driven current"
            ,),
        )
    return DqlSourceAssessment(
        model, "unknown", None, "unknown", None, False, False, "unknown",
        "not_reported", "unknown", False, "unsupported D_ql source model",
        ("documented D_ql provider",),
    )


def internal_dql_metadata(config: SimulationConfig) -> dict[str, Any]:
    """Metadata stored next to an internally generated GENRAY-like Dql."""

    assessment = assess_dql_source(config)
    return {
        "schema": "ecstartup.dql.v1",
        "generator": assessment.generator,
        "coordinate_basis": assessment.coordinate_basis,
        "requires_closed_flux_surfaces": assessment.requires_closed_flux_surfaces,
        "pre_cfs_compatible": assessment.pre_cfs_compatible,
        "power_normalized": assessment.power_normalized,
        "wave_branch": assessment.wave_branch,
        "conversion_evidence": assessment.mode_conversion_status,
        "evidence_level": assessment.evidence_level,
        "quantitative_eligible": assessment.quantitative_eligible,
    }


__all__ = ["DqlSourceAssessment", "assess_dql_source", "internal_dql_metadata"]
