"""Contracts for the small reference outputs published with the repository."""

from __future__ import annotations

import json
import math
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "examples" / "results"


def _load(name: str) -> dict[str, object]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_published_minimal_global_summary_is_a_completed_smoke_run() -> None:
    payload = _load("minimal_global_summary.json")
    final = payload["final"]
    assert payload["success"] is True
    assert payload["n_samples"] == 101
    assert payload["final_stage"] == "vacuum"
    assert 0.0 < final["Ip_A"] < 2.0
    assert final["field_configuration"] == "circular"


def test_published_fp_summary_retains_conservation_and_scope_warning() -> None:
    payload = _load("fp_synthetic_summary.json")
    final = payload["final_moments"]
    assessment = payload["dql_assessment"]
    assert payload["backend"] == "reduced_0d2v"
    assert payload["steps"] == 200
    assert math.isfinite(final["current_parallel_A"])
    assert payload["max_abs_particle_balance_relative"] < 1.0e-12
    assert assessment["evidence_level"] == "sensitivity_only"
    assert assessment["quantitative_eligible"] is False


def test_published_oxb_summary_does_not_overclaim_mode_conversion() -> None:
    payload = _load("oxb_candidate_summary.json")
    assert payload["available"] is True
    assert payload["maximum_density_m3"] > payload["O_cutoff_density_m3"]
    assert payload["ebwcd_assessment"] == "geometric_candidate_only"
    assert payload["xb_efficiency_estimate"] is None
    assert payload["combined_efficiency_estimate"] is None


def test_published_fieldline_summary_contains_tpc_and_vfc_components() -> None:
    payload = _load("fieldline_topology_summary.json")
    field = payload["field_diagnostics"]
    assert payload["total_sample_count"] == 48
    assert payload["valid_sample_count"] == 48
    assert field["field_provider"] == "composite_analytic_coils"
    assert {field["coil_0_role"], field["coil_1_role"]} == {"TPC", "VFC"}
    assert 0.0 <= payload["closed_fraction"] <= 1.0
