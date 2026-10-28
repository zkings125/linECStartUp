from __future__ import annotations

from ecstartup.config import load_config
from ecstartup.device_prediction_quality import assess_device_prediction_quality


def test_default_exl_config_is_diagnostic_only() -> None:
    gate = assess_device_prediction_quality(load_config("configs/linec_parameters.ini"))
    assert gate.prediction_level == "diagnostic_only"
    assert not gate.startup_decision_authorized
    assert not gate.current_prediction_authorized
    assert gate.missing_inputs


def test_complete_synthetic_gate_authorizes_both_predictions() -> None:
    config = load_config("configs/linec_parameters.ini")
    raw = config.model_dump(mode="python")
    raw["validation"].update(
        {
            "intended_use": "quantitative_device_prediction",
            "shot_resolved_coil_waveforms": True,
            "measured_pressure_waveform": True,
            "measured_zeff_or_impurity": True,
            "calibrated_rf_beam": True,
            "validated_power_normalized_dql": True,
            "wall_limiter_geometry": True,
            "orbit_loss_calibrated": True,
            "validation_success_shots": 3,
            "validation_failure_shots": 3,
            "validation_current_trace_shots": 3,
            "startup_sensitivity": 0.95,
            "startup_specificity": 0.95,
            "ip_relative_error": 0.10,
        }
    )
    complete = type(config).model_validate(raw)
    gate = assess_device_prediction_quality(complete)
    assert gate.prediction_level == "validated_device_prediction"
    assert gate.startup_decision_authorized
    assert gate.current_prediction_authorized
