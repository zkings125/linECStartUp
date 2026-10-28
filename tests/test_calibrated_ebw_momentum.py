from __future__ import annotations

import pytest

from ecstartup.config import GeometryConfig, RFConfig
from ecstartup.physics.calibrated_ebw_momentum import CalibratedEBWSettings


def test_ramp_gain_uses_cumulative_field_progress() -> None:
    settings = CalibratedEBWSettings(
        current_per_absorbed_power_A_per_W=0.2,
        ramp_gain_max=2.5,
        ramp_start_field_T=0.006,
        ramp_end_field_T=0.014,
        vertical_field_waveform=((0.0, 0.006), (0.02, 0.006), (0.06, 0.014), (0.10, 0.014)),
        ramp_enabled=True,
    )
    assert settings.ramp_gain(0.006) == pytest.approx(1.0)
    assert settings.ramp_gain(0.010) == pytest.approx(1.75)
    assert settings.ramp_gain(0.014) == pytest.approx(2.5)
    assert settings.ramp_rate_T_per_s(0.04) == pytest.approx(0.2)
    assert settings.ramp_rate_T_per_s(0.08) == pytest.approx(0.0)


def test_disabled_ramp_is_exactly_unity() -> None:
    settings = CalibratedEBWSettings(
        current_per_absorbed_power_A_per_W=0.2,
        ramp_gain_max=7.0,
        ramp_start_field_T=0.006,
        ramp_end_field_T=0.014,
        ramp_enabled=False,
    )
    assert settings.ramp_gain(0.014) == 1.0
    assert settings.ramp_progress(0.014) == 0.0


def test_invalid_calibration_coefficients_are_rejected() -> None:
    with pytest.raises(ValueError):
        CalibratedEBWSettings(current_per_absorbed_power_A_per_W=-0.1)
    with pytest.raises(ValueError):
        CalibratedEBWSettings(current_per_absorbed_power_A_per_W=0.1, ramp_gain_max=0.9)


def test_settings_accept_project_geometry_and_rf_types() -> None:
    # Smoke-check the public types imported by the calibrated module.
    geometry = GeometryConfig(
        R0_m=0.85,
        minor_radius_m=0.65,
        B_t_T=0.5,
        R_bounds_m=(0.15, 1.55),
        Z_bounds_m=(-1.0, 1.0),
    )
    rf = RFConfig()
    assert geometry.R0_m == pytest.approx(0.85)
    assert rf.current_drive_A_per_W == pytest.approx(0.0)
