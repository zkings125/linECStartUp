from __future__ import annotations

from pathlib import Path

from ecstartup.config import load_config


CONFIG_DIR = Path("configs/devices")


def test_cross_device_inputs_are_standalone_and_diagnostic_only() -> None:
    files = sorted(CONFIG_DIR.glob("*.ini"))
    assert len(files) == 7
    for path in files:
        config = load_config(path)
        assert config.validation.intended_use == "diagnostic_only"
        assert config.numerics.t_end_s > config.numerics.t_start_s
        resolved_window = config.startup_window.describe(
            t_start_s=config.numerics.t_start_s,
            t_end_s=config.numerics.t_end_s,
        )
        assert resolved_window["universal_duration_assumed"] is False
        assert len(config.rf.sources) == 1
        assert len(config.threshold_scan.ec_powers_W) == 6
        assert config.threshold_scan.ec_powers_W[0] == 0.0
        assert config.geometry.R0_m / config.geometry.minor_radius_m > 1.0


def test_device_set_prioritizes_spherical_tori_and_separates_cs_cases() -> None:
    configs = {load_config(path).validation.device_name: load_config(path) for path in CONFIG_DIR.glob("*.ini")}
    assert {"MAST", "SUNIST", "LATE", "QUEST", "VEST", "DIII-D", "J-TEXT"} == set(configs)
    assert configs["VEST"].geometry.field_configuration == "TPC"
    assert configs["VEST"].circuit.cs_mode == "with_cs"
    assert configs["DIII-D"].circuit.cs_mode == "with_cs"
    for name in ("MAST", "SUNIST", "LATE", "QUEST", "J-TEXT"):
        assert configs[name].circuit.cs_mode == "without_cs"
