from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecstartup.cli import main
from ecstartup.config import SimulationConfig, load_config
from ecstartup.kinetic_workflow import make_power_normalized_diffusion, run_kinetic
from ecstartup.physics.fokker_planck import VelocityGrid


def small_config(**kinetic_updates) -> SimulationConfig:
    kinetic = {
        "n_v_parallel": 31,
        "n_v_perp": 16,
        "v_parallel_max_m_s": 1.0e7,
        "v_perp_max_m_s": 1.0e7,
        "time_step_s": 1.0e-9,
        "steps": 2,
        "collision_frequency_s": 0.0,
    }
    kinetic.update(kinetic_updates)
    return SimulationConfig.model_validate(
        {
            "initial": {"ne_m3": 1.0e16, "ni_m3": 1.0e16, "Te_eV": 10.0},
            "kinetic": kinetic,
        }
    )


def test_workflow_reports_current_from_distribution() -> None:
    result = run_kinetic(small_config(electric_field_V_m=2.0))
    summary = result.summary()
    assert result.final_moments.current_parallel_A > 0.0
    assert summary["equilibrium_model"].startswith("none")
    assert summary["rf_diffusion_source"] == "disabled"
    assert summary["max_abs_particle_balance_relative"] < 1.0e-12


def test_external_npz_diffusion_contract(tmp_path: Path) -> None:
    config = small_config(rf_diffusion_model="external_npz", rf_diffusion_file=str(tmp_path / "dql.npz"))
    shape = (config.kinetic.n_v_parallel, config.kinetic.n_v_perp)
    np.savez(
        config.kinetic.rf_diffusion_file,
        D_parallel_m2_s3=np.zeros(shape),
        D_perp_m2_s3=np.zeros(shape),
    )
    result = run_kinetic(config)
    assert result.diffusion_source.startswith("external_npz:")


def test_external_npz_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, D_parallel_m2_s3=np.zeros((2, 2)), D_perp_m2_s3=np.zeros((2, 2)))
    config = small_config(rf_diffusion_model="external_npz", rf_diffusion_file=str(path))
    with pytest.raises(ValueError, match="match the velocity grid"):
        run_kinetic(config)


def test_genray_npz_is_scaled_by_absorbed_power(tmp_path: Path) -> None:
    path = tmp_path / "genray_dql.npz"
    config = small_config(
        rf_diffusion_model="genray_npz",
        rf_diffusion_file=str(path),
    )
    shape = (config.kinetic.n_v_parallel, config.kinetic.n_v_perp)
    np.savez(
        path,
        D_parallel_m2_s3=np.full(shape, 4.0),
        D_perp_m2_s3=np.full(shape, 2.0),
        reference_absorbed_power_W=np.asarray(2.0e4),
    )
    grid = VelocityGrid.uniform(
        config.kinetic.n_v_parallel,
        config.kinetic.n_v_perp,
        config.kinetic.v_parallel_max_m_s,
        config.kinetic.v_perp_max_m_s,
    )
    diffusion, reference, applied = make_power_normalized_diffusion(
        config, grid, absorbed_power_W=5.0e3
    )
    assert reference == pytest.approx(2.0e4)
    assert applied == pytest.approx(5.0e3)
    assert np.allclose(diffusion.D_parallel_m2_s3, 1.0)
    assert diffusion.source.startswith("genray_npz:")


def test_ini_and_cli_solve_fp(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "fp.ini"
    summary_path = tmp_path / "summary.json"
    distribution_path = tmp_path / "distribution.npz"
    input_path.write_text(
        """
[initial]
ne_m3 = 1.0e16
ni_m3 = 1.0e16
Te_eV = 10.0
[kinetic]
switchOn = 'disabled'
backend = 'reduced_0d2v'
n_v_parallel = 31
n_v_perp = 16
v_parallel_max_m_s = 1.0e7
v_perp_max_m_s = 1.0e7
time_step_s = 1.0e-9
steps = 2
collision_frequency_s = 0.0
electric_field_V_m = 1.0
rf_diffusion_model = 'zero'
""".strip(),
        encoding="utf-8",
    )
    parsed = load_config(input_path)
    assert parsed.kinetic.enabled is False
    main(
        [
            "solve-fp",
            str(input_path),
            "--output",
            str(summary_path),
            "--distribution-output",
            str(distribution_path),
        ]
    )
    capsys.readouterr()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["kinetic_switch_enabled"] is False
    assert summary["explicit_command_runs_when_disabled"] is True
    assert summary["final_moments"]["current_parallel_A"] > 0.0
    assert distribution_path.exists()
