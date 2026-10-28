from __future__ import annotations

import numpy as np
import pytest

from ecstartup.config import SimulationConfig
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.solver import integrate


def feedback_config(power_W: float) -> SimulationConfig:
    return SimulationConfig(
        geometry={"R0_m": 0.5, "minor_radius_m": 0.25},
        initial={"ne_m3": 1.0e17, "ni_m3": 1.0e17, "Te_eV": 20.0},
        rf={
            "enabled": True,
            "power_W": power_W,
            "absorption_fraction": 1.0,
            "current_drive_A_per_W": 999.0,
            "hot_fraction": 0.0,
            "pulse": {"start_s": 0.0, "end_s": 1.0e-3},
        },
        kinetic={
            "enabled": True,
            "coupling_mode": "moment_feedback",
            "n_v_parallel": 21,
            "n_v_perp": 12,
            "v_parallel_max_m_s": 8.0e6,
            "v_perp_max_m_s": 8.0e6,
            "time_step_s": 1.0e-8,
            "steps": 8,
            "collision_frequency_s": 2.0e5,
            "collision_temperature_eV": 20.0,
            "rf_diffusion_model": "synthetic_gaussian",
            "rf_diffusion_parallel_peak_m2_s3": 1.0e19,
            "rf_resonant_v_parallel_m_s": -2.0e6,
            "rf_resonant_v_perp_m_s": 2.0e6,
            "rf_sigma_v_parallel_m_s": 1.0e6,
            "rf_sigma_v_perp_m_s": 1.0e6,
            "feedback_reference_absorbed_power_W": 1.0e5,
            "feedback_power_fractions": [0.0, 0.5, 1.0],
            "feedback_replace_direct_rf": True,
        },
        circuit={"cs_mode": "without_cs"},
    )


def test_fp_moment_feedback_replaces_empirical_rf_current() -> None:
    model = GlobalStartupModel(feedback_config(1.0e5))
    evaluation = model.evaluate(5.0e-4, model.initial_state())
    assert evaluation.values["current_drive_model"] == "fp_moment_response_table"
    assert evaluation.values["I_rf_A"] == pytest.approx(evaluation.values["I_kinetic_A"])
    assert abs(float(evaluation.values["I_kinetic_A"])) > 0.0
    assert abs(float(evaluation.values["I_rf_A"])) < 999.0 * 1.0e5


def test_fp_feedback_is_zero_when_absorbed_power_is_zero() -> None:
    model = GlobalStartupModel(feedback_config(0.0))
    evaluation = model.evaluate(5.0e-4, model.initial_state())
    assert evaluation.values["I_kinetic_A"] == pytest.approx(0.0, abs=1.0e-12)
    assert evaluation.values["I_rf_A"] == pytest.approx(0.0, abs=1.0e-12)


def test_time_dependent_feedback_returns_distribution_history() -> None:
    raw = feedback_config(1.0e5).model_dump(mode="python")
    raw["kinetic"].update(
        {
            "coupling_mode": "time_dependent_feedback",
            "time_dependent_iterations": 2,
            "time_dependent_relaxation": 1.0,
            "time_dependent_snapshot_count": 3,
        }
    )
    raw["numerics"].update({"t_end_s": 6.0e-5, "dt_s": 2.0e-5})
    raw["outputs"].update(
        {"cadence_s": 2.0e-5, "write_csv": False, "write_netcdf": False}
    )
    result = integrate(SimulationConfig.model_validate(raw))
    assert result.success
    assert result.kinetic_history is not None
    assert result.kinetic_history.snapshot_f_e.shape[0] == 3
    assert any(
        row["current_drive_model"] == "fp_time_dependent_fixed_point"
        for row in result.diagnostics
    )
    assert np.max(np.abs(result.kinetic_history.incremental_current_A)) > 0.0
