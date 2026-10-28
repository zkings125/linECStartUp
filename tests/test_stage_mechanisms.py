from __future__ import annotations

import numpy as np

from ecstartup.config import SimulationConfig
from ecstartup.current_profile import MechanismCurrentEvolution
from ecstartup.solver import RunResult
from ecstartup.stage_mechanisms import diagnostic_stage_windows, summarize_stage_mechanisms


def _result(ne: list[float], te: list[float], ip: list[float]) -> RunResult:
    times = np.arange(len(ne), dtype=float) * 1.0e-3
    rows = [
        {"ne_m3": n, "Te_eV": t, "Ip_A": current, "stage": "vacuum"}
        for n, t, current in zip(ne, te, ip)
    ]
    return RunResult(times, [], rows, [], True, "synthetic")


def test_windows_follow_density_temperature_and_peak_boundaries() -> None:
    config = SimulationConfig(
        thresholds={
            "breakdown_density_m3": 1.0e16,
            "preionization_density_m3": 1.0e18,
            "preionization_Te_eV": 2.0,
        }
    )
    result = _result(
        [1.0e14, 2.0e16, 1.0e18, 2.0e18, 1.0e17],
        [0.2, 1.0, 3.0, 4.0, 2.0],
        [0.0, 1.0, 3.0, 5.0, 4.0],
    )
    windows = diagnostic_stage_windows(result, config)
    assert [(item["start_index"], item["end_index"]) for item in windows] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]


def test_mechanism_summary_excludes_residual_and_handles_unreached_stages() -> None:
    config = SimulationConfig()
    result = _result([1.0e14] * 4, [0.2] * 4, [0.0, 1.0, 2.0, 3.0])
    currents = MechanismCurrentEvolution(
        times_s=result.times_s,
        Ip_A=np.asarray([0.0, 1.0, 2.0, 3.0]),
        I_cs_A=np.zeros(4),
        I_rf_A=np.asarray([0.0, 1.0, 2.0, 3.0]),
        I_tail_A=np.asarray([0.0, 0.1, 0.1, 0.1]),
        I_vertical_A=np.zeros(4),
        I_initial_A=np.zeros(4),
        I_closure_residual_A=np.full(4, 1.0e9),
    )
    summaries = summarize_stage_mechanisms(result, config, currents)
    assert len(summaries) == 4
    assert summaries[0]["dominant_mechanism"] == "fp_ec_moment"
    assert summaries[1]["dominant_mechanism"] == "not reached"
    assert "closure_residual" not in summaries[0]["mechanism_fraction"]
