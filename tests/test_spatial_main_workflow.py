from __future__ import annotations

import numpy as np

from ecstartup.config import SimulationConfig
from ecstartup.solver import integrate


def test_open_field_1d2v_is_wired_into_bidirectional_main_workflow(tmp_path) -> None:
    fieldline = tmp_path / "fieldline.npz"
    np.savez_compressed(
        fieldline,
        s_m=np.linspace(0.0, 0.2, 3),
        R_m=np.linspace(0.5, 0.6, 3),
        Z_m=np.zeros(3),
        B_T=np.full(3, 0.8),
        cross_section_area_m2=np.full(3, 0.01),
    )
    config = SimulationConfig(
        rf={"enabled": False, "power_W": 0.0},
        kinetic={
            "enabled": True,
            "backend": "open_field_line_1d2v",
            "coupling_mode": "time_dependent_feedback",
            "coordinate_system": "velocity",
            "n_v_parallel": 5,
            "n_v_perp": 4,
            "v_parallel_max_m_s": 2.0e4,
            "v_perp_max_m_s": 2.0e4,
            "spatial_fieldline_file": str(fieldline),
            "spatial_sheath_model": "quasineutral_ambipolar",
            "time_dependent_iterations": 1,
            "time_dependent_snapshot_count": 2,
        },
        numerics={"t_end_s": 2.0e-8, "dt_s": 1.0e-8},
        outputs={"cadence_s": 1.0e-8},
    )
    result = integrate(config)
    assert result.success
    assert result.kinetic_history is not None
    assert result.kinetic_history.coupling_definition.startswith("global trajectory")
    assert len(result.kinetic_history.ionization_rate_m3_s) == len(result.times_s)
    assert result.diagnostics[-1]["kinetic_particle_power_closure"] == (
        "open_field_line_1d2v_replacement"
    )
