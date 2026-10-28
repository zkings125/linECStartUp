"""Device-level prediction quality gate.

The startup solver can always emit diagnostic and sensitivity quantities.
Calling those quantities a device prediction requires a separate evidence
decision: shot-resolved inputs, successful and failed startup validation, and
current-trace validation.  This module makes that decision deterministic and
machine readable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import QuantitativeValidationConfig, SimulationConfig


_PHYSICS_INPUTS = {
    "shot_resolved_coil_waveforms": "shot-resolved PF/TF/CS/TPC/VFC waveforms",
    "measured_pressure_waveform": "measured fill-pressure/gas waveform",
    "measured_zeff_or_impurity": "measured Zeff or impurity evolution",
    "calibrated_rf_beam": "calibrated EC launch, polarization and power waveform",
    "validated_power_normalized_dql": "validated power-normalized Dql",
    "wall_limiter_geometry": "device wall/limiter geometry",
    "orbit_loss_calibrated": "field-line/orbit-loss calibration",
}


@dataclass(frozen=True)
class DevicePredictionQuality:
    device_name: str
    shot_id: str | None
    intended_use: str
    prediction_level: str
    physics_inputs_complete: bool
    validation_dataset_complete: bool
    startup_metrics_pass: bool
    current_metric_pass: bool
    startup_decision_authorized: bool
    current_prediction_authorized: bool
    missing_inputs: tuple[str, ...]
    failed_metrics: tuple[str, ...]
    allowed_outputs: tuple[str, ...]
    blocked_outputs: tuple[str, ...]
    interpretation: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("missing_inputs", "failed_metrics", "allowed_outputs", "blocked_outputs"):
            payload[key] = list(payload[key])
        return payload


def _metric_failures(config: QuantitativeValidationConfig) -> list[str]:
    failures: list[str] = []
    if config.validation_success_shots < config.minimum_success_shots:
        failures.append(
            f"startup-success shots {config.validation_success_shots} < {config.minimum_success_shots}"
        )
    if config.validation_failure_shots < config.minimum_failure_shots:
        failures.append(
            f"startup-failure shots {config.validation_failure_shots} < {config.minimum_failure_shots}"
        )
    if config.startup_sensitivity is None:
        failures.append("startup sensitivity is not measured")
    elif config.startup_sensitivity < config.minimum_sensitivity:
        failures.append(
            f"startup sensitivity {config.startup_sensitivity:.3f} < {config.minimum_sensitivity:.3f}"
        )
    if config.startup_specificity is None:
        failures.append("startup specificity is not measured")
    elif config.startup_specificity < config.minimum_specificity:
        failures.append(
            f"startup specificity {config.startup_specificity:.3f} < {config.minimum_specificity:.3f}"
        )
    if config.validation_current_trace_shots < config.minimum_current_trace_shots:
        failures.append(
            "validated Ip traces "
            f"{config.validation_current_trace_shots} < {config.minimum_current_trace_shots}"
        )
    if config.ip_relative_error is None:
        failures.append("Ip relative error is not measured")
    elif config.ip_relative_error > config.maximum_ip_relative_error:
        failures.append(
            f"Ip relative error {config.ip_relative_error:.3f} > {config.maximum_ip_relative_error:.3f}"
        )
    return failures


def assess_device_prediction_quality(
    config: SimulationConfig | QuantitativeValidationConfig,
) -> DevicePredictionQuality:
    """Assess whether startup classification and Ip prediction may be claimed."""

    gate = config.validation if isinstance(config, SimulationConfig) else config
    missing = tuple(label for field, label in _PHYSICS_INPUTS.items() if not getattr(gate, field))
    physics_complete = not missing
    failures = _metric_failures(gate)
    validation_dataset_complete = (
        gate.validation_success_shots >= gate.minimum_success_shots
        and gate.validation_failure_shots >= gate.minimum_failure_shots
        and gate.validation_current_trace_shots >= gate.minimum_current_trace_shots
    )
    startup_metrics_pass = (
        gate.startup_sensitivity is not None
        and gate.startup_specificity is not None
        and gate.startup_sensitivity >= gate.minimum_sensitivity
        and gate.startup_specificity >= gate.minimum_specificity
        and gate.validation_success_shots >= gate.minimum_success_shots
        and gate.validation_failure_shots >= gate.minimum_failure_shots
    )
    current_metric_pass = (
        gate.ip_relative_error is not None
        and gate.ip_relative_error <= gate.maximum_ip_relative_error
        and gate.validation_current_trace_shots >= gate.minimum_current_trace_shots
    )
    quantitative_requested = gate.intended_use == "quantitative_device_prediction"
    startup_authorized = quantitative_requested and physics_complete and startup_metrics_pass
    current_authorized = startup_authorized and current_metric_pass

    if current_authorized:
        level = "validated_device_prediction"
    elif quantitative_requested and physics_complete and validation_dataset_complete:
        level = "quantitative_candidate"
    elif physics_complete:
        level = "calibration_ready"
    else:
        level = "diagnostic_only"

    allowed = [
        "EC resonance and accessibility diagnostics",
        "breakdown/preionization/CFS-stage diagnostics",
        "mechanism and parameter sensitivity",
        "surrogate Ip and threshold brackets labelled as unvalidated",
    ]
    blocked: list[str] = []
    if startup_authorized:
        allowed.append("device-level startup success/failure classification")
    else:
        blocked.append("device-level startup success/failure classification")
    if current_authorized:
        allowed.append("device-level Ip(t) prediction within the validated error envelope")
    else:
        blocked.append("device-level Ip(t) prediction or a precise current value")

    if current_authorized:
        interpretation = (
            "The configured evidence authorizes device-level startup classification and Ip(t) "
            "prediction only inside the validated parameter domain."
        )
    else:
        interpretation = (
            "The run is diagnostic/calibration evidence. Startup scores, threshold brackets and Ip "
            "are not validated device predictions and must not be used as precise cross-device answers."
        )

    return DevicePredictionQuality(
        device_name=gate.device_name,
        shot_id=gate.shot_id,
        intended_use=gate.intended_use,
        prediction_level=level,
        physics_inputs_complete=physics_complete,
        validation_dataset_complete=validation_dataset_complete,
        startup_metrics_pass=startup_metrics_pass,
        current_metric_pass=current_metric_pass,
        startup_decision_authorized=startup_authorized,
        current_prediction_authorized=current_authorized,
        missing_inputs=missing,
        failed_metrics=tuple(failures),
        allowed_outputs=tuple(allowed),
        blocked_outputs=tuple(blocked),
        interpretation=interpretation,
    )
