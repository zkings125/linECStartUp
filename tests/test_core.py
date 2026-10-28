from dataclasses import replace

import numpy as np
import pytest

from ecstartup.config import GeometryConfig, SimulationConfig
from ecstartup.diagnostics import StageManager
from ecstartup.physics.magnetic import CircularMagneticField, TPCMagneticField, VFCMagneticField, make_magnetic_field
from ecstartup.physics.open_field_loss import OpenFieldLossModel
from ecstartup.physics.rates import CollisionRateModel
from ecstartup.physics.topology import SurrogateTopologyDetector
from ecstartup.physics.tabulated_rates import TabulatedRateModel
from ecstartup.comparison import config_with_override
from ecstartup.state import StartupStage
from ecstartup.workflow import run_simulation
from ecstartup.waveform import linear_value


def test_geometry_volume_is_positive():
    geometry = GeometryConfig()
    assert geometry.resolved_volume_m3 > 0.0
    assert geometry.resolved_wall_area_m2 > 0.0


def test_unknown_config_field_is_rejected():
    with pytest.raises(Exception):
        SimulationConfig.model_validate({"not_a_real_field": 1})


def test_rates_are_nonnegative_and_thresholded():
    gas = SimulationConfig().gas
    model = CollisionRateModel(gas)
    cold = model.evaluate(0.01, 1.0e15, 1.0e19, 100.0)
    warm = model.evaluate(20.0, 1.0e15, 1.0e19, 100.0)
    assert cold.K_ion_m3_s >= 0.0
    assert warm.K_ion_m3_s > cold.K_ion_m3_s
    assert warm.nu_ei_s > 0.0


def test_topology_monotonic_in_current():
    detector = SurrogateTopologyDetector(GeometryConfig())
    low = detector.detect(0.0)
    high = detector.detect(250_000.0)
    assert low.closed_fraction == 0.0
    assert high.closed_fraction >= low.closed_fraction


def test_magnetic_field_scales_with_current():
    field = CircularMagneticField(GeometryConfig())
    b1 = field.evaluate(0.75, 0.0, 50_000.0)
    b2 = field.evaluate(0.75, 0.0, 100_000.0)
    assert abs(b2.B_Z_T - field.geometry.B_vertical_T) == pytest.approx(
        2.0 * abs(b1.B_Z_T - field.geometry.B_vertical_T)
    )


def test_field_configuration_providers_are_distinct_and_factory_selected():
    tpc_geometry = GeometryConfig(field_configuration="TPC", field_configuration_scale=2.0)
    vfc_geometry = GeometryConfig(field_configuration="VFC", field_configuration_scale=2.0)
    tpc = make_magnetic_field(tpc_geometry)
    vfc = make_magnetic_field(vfc_geometry)
    assert isinstance(tpc, TPCMagneticField)
    assert isinstance(vfc, VFCMagneticField)
    assert tpc.configuration == "TPC"
    assert vfc.configuration == "VFC"
    assert tpc.diagnostics(100_000.0)["Bp_edge_provider_T"] == pytest.approx(
        2.0 * CircularMagneticField(tpc_geometry).diagnostics(100_000.0)["Bp_edge_provider_T"]
    )
    assert vfc.diagnostics(100_000.0)["B_vertical_provider_T"] == pytest.approx(
        2.0 * GeometryConfig().B_vertical_T
    )


def test_tpc_decay_index_changes_midradius_field_diagnostic():
    """The TPC decay control must affect a field value, not only echo input."""

    common = {
        "field_configuration": "TPC",
        "tpc_mirror_ratio": 2.0,
        "tpc_trapped_fraction": 0.5,
    }
    shallow = TPCMagneticField(GeometryConfig(**common, tpc_decay_index=0.0))
    steep = TPCMagneticField(GeometryConfig(**common, tpc_decay_index=4.0))
    ip = 100_000.0
    r_mid = shallow.geometry.R0_m + 0.5 * shallow.geometry.minor_radius_m
    shallow_diag = shallow.diagnostics(ip)
    steep_diag = steep.diagnostics(ip)
    assert shallow_diag["tpc_B_phi_midradius_T"] != pytest.approx(
        steep_diag["tpc_B_phi_midradius_T"]
    )
    assert shallow_diag["tpc_mirror_shape_midradius"] > steep_diag["tpc_mirror_shape_midradius"]
    assert shallow.evaluate(r_mid, 0.0, ip).B_phi_T != pytest.approx(
        steep.evaluate(r_mid, 0.0, ip).B_phi_T
    )


def test_vfc_decay_index_changes_edge_field_diagnostic():
    """The VFC decay index must modify the radial vertical-field profile."""

    common = {"field_configuration": "VFC", "B_vertical_T": 0.02}
    flat = VFCMagneticField(GeometryConfig(**common, vfc_decay_index=0.0))
    decaying = VFCMagneticField(GeometryConfig(**common, vfc_decay_index=2.0))
    ip = 100_000.0
    r_edge = flat.geometry.R0_m + flat.geometry.minor_radius_m
    flat_diag = flat.diagnostics(ip)
    decaying_diag = decaying.diagnostics(ip)
    assert flat_diag["vfc_vertical_field_edge_T"] != pytest.approx(
        decaying_diag["vfc_vertical_field_edge_T"]
    )
    assert flat.evaluate(r_edge, 0.0, ip).B_Z_T != pytest.approx(
        decaying.evaluate(r_edge, 0.0, ip).B_Z_T
    )


def test_zero_vertical_field_gives_exactly_zero_charge_separation_current():
    config = SimulationConfig(
        geometry={"B_vertical_T": 0.0, "vfc_current_fraction": 0.2},
        initial={"n_hot_m3": 1.0e17, "hot_energy_J_m3": 1.0e3},
    )
    model = __import__("ecstartup.physics.global_model", fromlist=["GlobalStartupModel"]).GlobalStartupModel(config)
    evaluation = model.evaluate(0.01, model.initial_state())
    assert evaluation.values["B_vertical_T"] == 0.0
    assert evaluation.circuit.I_vertical_charge_A == 0.0
    assert evaluation.values["I_vertical_charge_A"] == 0.0


def test_configuration_loss_factor_is_applied_once_with_effective_topology_length():
    """Passing an effective topology length must not multiply its factor again."""

    config = SimulationConfig(
        geometry={
            "field_configuration": "TPC",
            "tpc_mirror_ratio": 2.0,
            "tpc_trapped_fraction": 0.5,
            "closed_confinement_gain": 0.0,
        }
    )
    model = __import__("ecstartup.physics.global_model", fromlist=["GlobalStartupModel"]).GlobalStartupModel(config)
    field_state = model.magnetic.configuration_state(0.0)
    loss = OpenFieldLossModel(config.geometry, model.rates_model, config.initial.Ti_eV)
    state = model.initial_state()
    baseline = loss.evaluate(state, closed_fraction=0.0)
    direct_state = loss.evaluate(state, closed_fraction=0.0, magnetic_state=field_state)
    effective_length = config.geometry.connection_length_m * field_state.connection_length_factor
    topology_path = loss.evaluate(
        state,
        closed_fraction=0.0,
        connection_length_m=effective_length,
        magnetic_state=field_state,
    )
    factor = field_state.connection_length_factor
    assert direct_state.electron_s / baseline.electron_s == pytest.approx(factor)
    assert topology_path.electron_s == pytest.approx(direct_state.electron_s)
    assert topology_path.ion_s == pytest.approx(direct_state.ion_s)
    assert topology_path.configuration_multiplier == pytest.approx(1.0)


def test_current_components_expose_tpc_vfc_aliases():
    config = SimulationConfig(geometry={"vfc_current_fraction": 0.2})
    model = __import__("ecstartup.physics.global_model", fromlist=["GlobalStartupModel"]).GlobalStartupModel(config)
    state = replace(model.initial_state(), n_hot_m3=1.0e17, hot_energy_J_m3=1.0e3)
    terms = model.evaluate(0.01, state).circuit
    assert terms.I_tpc_A == pytest.approx(terms.I_orbit_A)
    assert terms.I_vfc_A != 0.0


def test_stage_manager_order():
    manager = StageManager(SimulationConfig().thresholds)
    base = {"ne_m3": 1.0e14, "Te_eV": 0.2, "net_growth_s": -1.0, "closed_fraction": 0.0}
    assert manager.stage == StartupStage.VACUUM
    for _ in range(3):
        event = manager.update(0.0, {**base, "ne_m3": 2.0e16, "net_growth_s": 1.0})
    assert event is not None and event.type == "breakdown"
    for _ in range(3):
        event = manager.update(0.0, {**base, "ne_m3": 2.0e18, "Te_eV": 3.0, "net_growth_s": 1.0})
    assert event is not None and event.type == "preionization"
    event = manager.update(0.0, {**base, "ne_m3": 2.0e18, "Te_eV": 3.0, "net_growth_s": 1.0, "closed_fraction": 0.1})
    assert event is not None and event.type == "cfs_candidate"


def test_short_run_is_finite():
    config = SimulationConfig(
        numerics={"t_end_s": 2.0e-5, "dt_s": 2.0e-6, "solver": "BDF"},
        outputs={"directory": "runs/test_unused", "write_netcdf": False, "write_csv": False},
    )
    result = run_simulation(config, output_directory=False)
    assert result.success
    assert len(result.states) > 1
    values = np.asarray([state.to_vector() for state in result.states])
    assert np.isfinite(values).all()
    assert (values[:, :4] >= 0.0).all()


def test_exact_vacuum_seed_is_allowed():
    config = SimulationConfig(
        initial={"ne_m3": 0.0, "ni_m3": 0.0, "Te_eV": 0.0, "Ti_eV": 0.0, "seed_rate_m3_s": 1.0e15},
        numerics={"t_end_s": 1.0e-6, "dt_s": 1.0e-6},
        outputs={"write_netcdf": False, "write_csv": False},
    )
    result = run_simulation(config, output_directory=False)
    assert result.success
    assert result.final_state.ne_m3 >= 0.0


def test_waveform_interpolation_and_validation():
    points = [(0.0, 0.0), (1.0, 2.0)]
    assert linear_value(points, 0.5, 99.0) == pytest.approx(1.0)
    assert linear_value(points, 2.0, 99.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        GeometryConfig(vertical_field_waveform=[(1.0, 0.0), (0.0, 1.0)])
    with pytest.raises(ValueError, match="有限数"):
        GeometryConfig(vertical_field_waveform=[(0.0, float("nan"))])


def test_tabulated_rate_provider():
    model = TabulatedRateModel(SimulationConfig().gas, "data/rates_demo.csv")
    rates = model.evaluate(10.0, 1.0e15, 1.0e19, 100.0)
    assert rates.K_ion_m3_s == pytest.approx(2.0e-14)
    assert rates.K_momentum_m3_s > 0.0


def test_tabulated_rate_provider_rejects_nonfinite_and_degenerate_tables(tmp_path):
    header = "Te_eV,EoverN_Td,K_ion_m3_s,K_exc_m3_s,K_att_m3_s,K_momentum_m3_s,nu_ei_s\n"
    rows = [
        "1,1,1,1,1,1,1\n",
        "2,1,1,1,1,1,1\n",
        "1,2,1,1,1,1,1\n",
    ]
    valid = tmp_path / "valid.csv"
    valid.write_text(header + "".join(rows), encoding="utf-8")
    # Duplicate coordinates are rejected before scipy's Qhull error leaks out.
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(header + "".join(rows[:2] + [rows[0], rows[2]]), encoding="utf-8")
    with pytest.raises(ValueError, match="不能重复"):
        TabulatedRateModel(SimulationConfig().gas, duplicate)

    collinear = tmp_path / "collinear.csv"
    collinear.write_text(header + "1,1,1,1,1,1,1\n2,2,1,1,1,1,1\n3,3,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不能共线"):
        TabulatedRateModel(SimulationConfig().gas, collinear)

    nonfinite = tmp_path / "nonfinite.csv"
    nonfinite.write_text(header + "1,1,nan,1,1,1,1\n2,1,1,1,1,1,1\n1,2,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN/Inf"):
        TabulatedRateModel(SimulationConfig().gas, nonfinite)


def test_device_override_rederives_geometry_length():
    config = config_with_override(SimulationConfig(), {"geometry": {"R0_m": 1.0, "R_bounds_m": [0.5, 1.5]}})
    assert config.circuit.plasma_length_m == pytest.approx(2.0 * np.pi)
