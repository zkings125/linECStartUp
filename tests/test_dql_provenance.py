from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecstartup.cli import main
from ecstartup.config import SimulationConfig
from ecstartup.dql_provenance import assess_dql_source


def _config(model: str = "zero") -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "rf": {
                "sources": [
                    {
                        "name": "EC28_OXB",
                        "frequency_Hz": 28.0e9,
                        "launch": {"mode": "O-X-B", "polarization": "O"},
                    }
                ]
            },
            "kinetic": {"rf_diffusion_model": model},
        }
    )


def test_zero_dql_is_explicitly_unavailable() -> None:
    result = assess_dql_source(_config())
    assert result.availability == "unavailable"
    assert result.pre_cfs_compatible
    assert not result.quantitative_eligible


def test_internal_dql_is_pre_cfs_but_does_not_confirm_ebw() -> None:
    result = assess_dql_source(_config("genray_like_internal"))
    assert result.pre_cfs_compatible
    assert result.requires_closed_flux_surfaces is False
    assert result.power_normalized
    assert result.wave_branch == "electromagnetic_ec_or_unresolved_oxb"
    assert result.mode_conversion_status == "geometric_candidate_not_fullwave"
    assert not result.quantitative_eligible


def test_verified_open_fieldline_fullwave_ebw_metadata_passes(tmp_path: Path) -> None:
    path = tmp_path / "dql.npz"
    metadata = {
        "schema": "ecstartup.fullwave_spatial_dql.v1",
        "coordinate_system": "relativistic_momentum",
        "target_absorbed_power_W": 1.0e5,
        "converged": True,
        "mesh_verified": True,
        "wave_branch": "ebw",
        "mode_decomposition_verified": True,
        "fullwave_power_balance_relative": 1.0e-5,
        "local_absorption_balance_relative": 2.0e-5,
        "power_budget": {"ebw_out_W": 7.0e4},
        "solver_name": "test-hot-fullwave",
    }
    np.savez(path, metadata_json=np.asarray(json.dumps(metadata)))
    result = assess_dql_source(_config(), source_file=path)
    assert result.pre_cfs_compatible
    assert result.wave_branch == "ebw"
    assert result.mode_conversion_status == "verified_fullwave"
    assert result.quantitative_eligible


def test_cli_assess_dql_reports_current_config(capsys) -> None:
    main(["assess-dql", "configs/devices/mast_ebw_startup.ini"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["availability"] == "generated"
    assert payload["wave_branch"] == "electromagnetic_ec_or_unresolved_oxb"
    assert payload["pre_cfs_compatible"]
    assert not payload["quantitative_eligible"]
