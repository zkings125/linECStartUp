from __future__ import annotations

from ecstartup.config import SimulationConfig
from ecstartup.threshold_scan import run_threshold_scan


def test_threshold_scan_uses_zero_power_baseline_and_is_power_ordered() -> None:
    config = SimulationConfig(
        initial={"ne_m3": 1.0e17, "ni_m3": 1.0e17, "Te_eV": 5.0},
        rf={
            "enabled": True,
            "power_W": 1.0e3,
            "absorption_fraction": 1.0,
            "current_drive_A_per_W": 0.1,
            "hot_fraction": 0.0,
            "pulse": {"start_s": 0.0, "end_s": 1.0e-3},
        },
        circuit={"cs_mode": "without_cs", "loop_voltage_V": 10.0},
        thresholds={"preionization_density_m3": 1.0e16, "preionization_Te_eV": 1.0},
        threshold_scan={
            "enabled": True,
            "pressures_Pa": [0.05],
            "effective_charges": [1.0],
            "ec_powers_W": [1.0e3, 5.0e3],
            "cs_modes": ["without_cs"],
            "success_increment_A": 1.0,
            "require_preionization": False,
            "write_pdf_report": False,
        },
        numerics={"t_end_s": 2.0e-5, "dt_s": 2.0e-6},
        outputs={"cadence_s": 2.0e-6, "write_netcdf": False, "write_csv": False},
    )
    payload = run_threshold_scan(config)
    cases = sorted(payload["cases"], key=lambda item: item["ec_power_W"])
    assert payload["criterion"]["baseline"].startswith("same pressure")
    assert cases[1]["peak_signed_ec_increment_A"] > cases[0]["peak_signed_ec_increment_A"]
    assert payload["thresholds"][0]["minimum_successful_ec_power_W"] is not None


def test_threshold_scan_adaptively_brackets_boundary() -> None:
    config = SimulationConfig(
        initial={"ne_m3": 1.0e17, "ni_m3": 1.0e17, "Te_eV": 5.0},
        rf={
            "enabled": True,
            "absorption_fraction": 1.0,
            "current_drive_A_per_W": 0.1,
            "hot_fraction": 0.0,
            "pulse": {"start_s": 0.0, "end_s": 1.0e-3},
        },
        circuit={"cs_mode": "without_cs"},
        threshold_scan={
            "enabled": True,
            "pressures_Pa": [0.05],
            "effective_charges": [1.0],
            "ec_powers_W": [1.0e3, 2.0e4],
            "cs_modes": ["without_cs"],
            "success_increment_A": 5.0,
            "require_preionization": False,
            "refine_enabled": True,
            "refine_power_tolerance_W": 500.0,
            "write_pdf_report": False,
        },
        numerics={"t_end_s": 2.0e-5, "dt_s": 2.0e-6},
        outputs={"cadence_s": 2.0e-6, "write_netcdf": False, "write_csv": False},
    )
    payload = run_threshold_scan(config)
    threshold = payload["thresholds"][0]
    assert threshold["refinement_iterations"] > 0
    assert threshold["minimum_successful_ec_power_W"] - threshold["last_failed_ec_power_W"] <= 500.0
    assert any(row["sample_kind"] == "refined" for row in payload["cases"])
