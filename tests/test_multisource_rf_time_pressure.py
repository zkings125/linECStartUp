from __future__ import annotations

import pytest

from ecstartup.config import SimulationConfig
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.solver import integrate


def multisource_config() -> SimulationConfig:
    return SimulationConfig(
        gas={"pressure_Pa": 0.05, "pressure_waveform_Pa": [[-0.01, 0.01], [0.0, 0.05], [0.01, 0.10]]},
        rf={
            "enabled": True,
            "sources": [
                {
                    "name": "EC28_O",
                    "frequency_Hz": 28.0e9,
                    "power_W": 100.0e3,
                    "absorption_fraction": 0.5,
                    "pulse": {"start_s": -0.01, "end_s": 0.005},
                    "launch": {"R_m": 1.606, "Z_m": 0.795, "mode": "O", "polarization": "O"},
                },
                {
                    "name": "EC50_X",
                    "frequency_Hz": 50.0e9,
                    "power_W": 150.0e3,
                    "absorption_fraction": 0.5,
                    "pulse": {"start_s": -0.005, "end_s": 0.01},
                    "launch": {"R_m": 1.725, "Z_m": 0.985, "mode": "X", "polarization": "X"},
                },
            ],
        },
        numerics={"t_start_s": -0.01, "t_end_s": 0.01, "dt_s": 0.001},
        outputs={"cadence_s": 0.001, "write_netcdf": False, "write_csv": False},
    )


def test_negative_time_grid_and_pressure_waveform() -> None:
    config = multisource_config()
    result = integrate(config)
    assert result.success
    assert result.times_s[0] == pytest.approx(-0.01)
    assert result.times_s[-1] == pytest.approx(0.01)
    assert result.diagnostics[0]["target_pressure_Pa"] == pytest.approx(0.01)
    assert result.diagnostics[-1]["target_pressure_Pa"] == pytest.approx(0.10)
    assert result.diagnostics[-1]["pL_Pa_m"] >= 0.0


def test_multisource_rf_is_independently_timed_and_summed() -> None:
    config = multisource_config()
    model = GlobalStartupModel(config)
    state = model.initial_state()
    both = model.rf_model.evaluate(0.0, state)
    first_only = model.rf_model.evaluate(-0.0075, state)
    assert len(both.components) == 2
    assert both.components[0].pulse_factor == pytest.approx(1.0)
    assert both.components[1].pulse_factor == pytest.approx(1.0)
    assert both.power_absorbed_W == pytest.approx(
        both.components[0].power_absorbed_W + both.components[1].power_absorbed_W
    )
    assert first_only.components[0].power_absorbed_W > 0.0
    assert first_only.components[1].power_absorbed_W == 0.0
    diagnostics = both.diagnostics()
    assert diagnostics["rf_0_name"] == "EC28_O"
    assert diagnostics["rf_1_name"] == "EC50_X"


def test_master_rf_switch_disables_all_sources() -> None:
    raw = multisource_config().model_dump(mode="python")
    raw["rf"]["enabled"] = False
    config = SimulationConfig.model_validate(raw)
    model = GlobalStartupModel(config)
    terms = model.rf_model.evaluate(0.0, model.initial_state())
    assert terms.power_absorbed_W == 0.0
    assert all(component.power_absorbed_W == 0.0 for component in terms.components)


@pytest.mark.parametrize(
    ("enabled_28", "enabled_50", "active_indices"),
    [
        (True, False, {0}),
        (False, True, {1}),
        (True, True, {0, 1}),
    ],
    ids=["28GHz_O_only", "50GHz_X_only", "28O_plus_50X"],
)
def test_28O_and_50X_sources_can_run_separately_or_together(
    enabled_28: bool,
    enabled_50: bool,
    active_indices: set[int],
) -> None:
    raw = multisource_config().model_dump(mode="python")
    raw["rf"]["sources"][0]["enabled"] = enabled_28
    raw["rf"]["sources"][1]["enabled"] = enabled_50
    config = SimulationConfig.model_validate(raw)
    model = GlobalStartupModel(config)
    terms = model.rf_model.evaluate(0.0, model.initial_state())

    positive_indices = {
        index for index, component in enumerate(terms.components) if component.power_absorbed_W > 0.0
    }
    assert positive_indices == active_indices
    assert terms.power_absorbed_W == pytest.approx(
        sum(component.power_absorbed_W for component in terms.components)
    )
