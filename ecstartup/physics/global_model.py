"""0D 全局启动模型：粒子、电子能量、热尾和电路的耦合。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np

from ..config import SimulationConfig
from ..constants import E_CHARGE_C, EPS0_F_M, K_B_J_K, M_E_KG
from ..state import GlobalState
from .charge_closure import ChargeClosureModel, ChargeClosureTerms
from .circuit import CircuitModel, CircuitTerms
from .magnetic import MagneticConfigurationState, MagneticFieldProvider, make_magnetic_field
from .open_field_loss import OpenFieldLossModel
from .rates import CollisionRateModel, RateCoefficients
from .rf import ECRHSource, RFSourceTerms
from .species_wall import DYONSpeciesWallModel, ExtendedGlobalState, SpeciesWallTerms
from .topology import SurrogateTopologyDetector, TopologyResult


@dataclass(frozen=True)
class PhysicsEvaluation:
    rates: RateCoefficients
    rf: RFSourceTerms
    circuit: CircuitTerms
    topology: TopologyResult
    values: Dict[str, float | str]
    species_wall: SpeciesWallTerms | None = None
    charge_closure: ChargeClosureTerms | None = None


class GlobalStartupModel:
    """可运行的 0D 原型。

    方程对应一个体平均控制体：

    * `dn_e/dt = S_ion + S_sec - S_rec - S_att - S_wall + S_seed`
    * `dn_i/dt = S_ion - S_rec - S_i,wall`
    * `dn_n/dt = -S_ion + S_rec + S_recycle - S_pump + S_feed`
    * `dU_e/dt = P_ohm + P_ECRH - P_ion - P_exc - P_rad - P_wall - P_particle`
    * `L_p dI_p/dt = V_loop - R_p I_p + R_p(I_RF + I_tail + I_vertical_charge)`

    反应率和轨道电流闭合均可替换；默认版本只用于验证软件耦合和数量级，
    不足以替代 1D-2V 或实验标定。
    """

    def __init__(
        self,
        config: SimulationConfig,
        rates_model=None,
        rf_model=None,
        topology=None,
        loss_model=None,
        *,
        magnetic_field: MagneticFieldProvider | None = None,
        current_drive=None,
        circuit_model: CircuitModel | None = None,
        species_wall_model: DYONSpeciesWallModel | None = None,
        charge_closure_model: ChargeClosureModel | None = None,
        kinetic_closure=None,
    ) -> None:
        """Build the coupled model from explicit, replaceable providers.

        The positional arguments preserve the v0.1 extension points.  New
        backends should be supplied through the keyword-only providers so a
        real coil/field-line model can replace the analytic magnetic field
        without changing the ODE or the workflow.
        """

        self.config = config
        self.geometry = config.geometry
        self.rates_model = rates_model if rates_model is not None else CollisionRateModel(config.gas)
        self.rf_model = rf_model if rf_model is not None else ECRHSource(config.rf, config.geometry)
        self.magnetic: MagneticFieldProvider = (
            magnetic_field if magnetic_field is not None else make_magnetic_field(config.geometry)
        )
        if current_drive is None and config.kinetic.coupling_mode == "moment_feedback":
            from .kinetic_feedback import KineticMomentCurrentDrive

            current_drive = KineticMomentCurrentDrive(config)
        self.circuit_model = circuit_model if circuit_model is not None else CircuitModel(
            config.circuit,
            config.geometry,
            config.rf,
            current_drive=current_drive,
        )
        self.topology = topology if topology is not None else SurrogateTopologyDetector(config.geometry, self.magnetic)
        self.loss_model = (
            loss_model
            if loss_model is not None
            else OpenFieldLossModel(config.geometry, self.rates_model, config.initial.Ti_eV)
        )
        self.volume_m3 = config.geometry.resolved_volume_m3
        self.wall_area_m2 = config.geometry.resolved_wall_area_m2
        self.species_wall_model = species_wall_model
        if self.species_wall_model is None and config.species_wall.enabled:
            atomic_rates = None
            if config.species_wall.atomic_backend == "openadas_adf11":
                from .adas import OpenADASAtomicRateProvider

                atomic_rates = OpenADASAtomicRateProvider(
                    config.species_wall.atomic_manifest,
                    bounds=config.species_wall.atomic_bounds,
                )
            self.species_wall_model = DYONSpeciesWallModel(
                config.species_wall,
                config.geometry,
                atomic_rates=atomic_rates,
            )
        self._has_fuel_species = any(item.role == "fuel" for item in config.species_wall.species)
        self.charge_closure_model = (
            charge_closure_model if charge_closure_model is not None else ChargeClosureModel(config.charge_closure)
        )
        self.kinetic_closure = kinetic_closure

    def _configuration_state(self, t_s: float, Ip_A: float) -> MagneticConfigurationState:
        provider = getattr(self.magnetic, "configuration_state", None)
        if callable(provider):
            return provider(Ip_A, t_s)
        return MagneticConfigurationState(configuration=getattr(self.magnetic, "configuration", "custom"))

    def _detect_topology(self, Ip_A: float, t_s: float) -> TopologyResult:
        try:
            return self.topology.detect(Ip_A, t_s)
        except TypeError as exc:
            # Preserve compatibility with a v0.1 custom detector that accepted
            # only Ip_A; do not swallow errors raised inside a new detector.
            if "positional argument" not in str(exc) and "takes" not in str(exc):
                raise
            return self.topology.detect(Ip_A)

    def _evaluate_losses(self, state: GlobalState, topology: TopologyResult, magnetic_state: MagneticConfigurationState):
        """Call old and new loss providers without hiding a custom backend error."""

        try:
            return self.loss_model.evaluate(
                state,
                topology.closed_fraction,
                connection_length_m=topology.connection_length_m,
                magnetic_state=magnetic_state,
            )
        except TypeError as exc:
            # Third-party v0.1 providers accepted only (state, closed_fraction).
            # Retry only when the signature truly lacks the new keywords.
            message = str(exc)
            if "unexpected keyword" not in message and "positional argument" not in message:
                raise
            return self.loss_model.evaluate(state, topology.closed_fraction)

    def initial_state(self) -> GlobalState | ExtendedGlobalState:
        cfg = self.config
        nn = cfg.initial.nn_m3
        if nn is None:
            nn = cfg.gas.pressure_at(cfg.numerics.t_start_s) / (K_B_J_K * cfg.gas.Tn_K)
        species_state = self.species_wall_model.initial_state() if self.species_wall_model is not None else None
        initial_ne = float(cfg.initial.ne_m3)
        if cfg.charge_closure.mode != "legacy" and cfg.charge_closure.initialize_ne_from_ions:
            initial_ne = float(cfg.initial.ni_m3)
            if species_state is not None:
                for species in self.species_wall_model.layout.species:
                    if species.role != "impurity":
                        continue
                    densities = species_state.charge_densities(species.symbol)
                    initial_ne += float(np.dot(np.arange(species.max_charge + 1), densities))
        ue = 1.5 * initial_ne * cfg.initial.Te_eV * E_CHARGE_C
        core = GlobalState(
            ne_m3=initial_ne,
            ni_m3=cfg.initial.ni_m3,
            nn_m3=nn,
            electron_energy_J_m3=ue,
            Ip_A=cfg.initial.Ip_A,
            n_hot_m3=cfg.initial.n_hot_m3,
            hot_energy_J_m3=cfg.initial.hot_energy_J_m3,
        )
        if self.species_wall_model is None:
            return core
        return ExtendedGlobalState(core, species_state)

    def state_from_vector(self, vector: np.ndarray) -> GlobalState | ExtendedGlobalState:
        """Deserialize the model-owned state layout for the generic solver."""

        if self.species_wall_model is None:
            return GlobalState.from_vector(vector)
        return ExtendedGlobalState.from_vector(self.species_wall_model.layout, vector)

    def absolute_tolerance_vector(self, scalar_floor: float = 1.0e-8) -> np.ndarray:
        """Return unit-aware absolute tolerances in the active state order."""

        core = np.asarray([1.0e5, 1.0e5, 1.0e10, 1.0e-8, 1.0e-3, 1.0e5, 1.0e-8])
        if self.species_wall_model is None:
            return np.maximum(core, scalar_floor)
        extension = []
        for name in self.species_wall_model.layout.vector_names:
            extension.append(1.0e3 if name.startswith("wall_inventory_") else 1.0e5)
        return np.maximum(np.concatenate((core, np.asarray(extension))), scalar_floor)

    def evaluate(self, t_s: float, state: GlobalState | ExtendedGlobalState) -> PhysicsEvaluation:
        extended_state = state.projected() if isinstance(state, ExtendedGlobalState) else None
        state = extended_state.core if extended_state is not None else state.projected()
        cfg = self.config
        nn = max(state.nn_m3, 0.0)
        ne = max(state.ne_m3, 0.0)
        ni = max(state.ni_m3, 0.0)
        # E/N 用 Td 表示：1 Td = 10^-21 V m^2。
        V_loop = self.circuit_model.loop_voltage(t_s)
        E_loop = V_loop / self.circuit_model.length_m
        EoverN_Td = abs(E_loop) / max(nn, 1.0) * 1.0e21
        rates = self.rates_model.evaluate(state.Te_eV, ne, nn, EoverN_Td)
        rf_terms = self.rf_model.evaluate(t_s, state)
        topology = self._detect_topology(state.Ip_A, t_s)
        magnetic_state = self._configuration_state(t_s, state.Ip_A)
        loss_times = self._evaluate_losses(state, topology, magnetic_state)
        effective_charge = 1.0
        if self.species_wall_model is not None and extended_state is not None:
            effective_charge = self.species_wall_model.effective_charge(
                extended_state.species_wall, state
            )
        if cfg.circuit.effective_charge_override is not None:
            effective_charge = float(cfg.circuit.effective_charge_override)
        circuit = self.circuit_model.evaluate(
            t_s,
            state,
            rates,
            rf_terms,
            state.Te_eV,
            magnetic_state=magnetic_state,
            loss_times=loss_times,
            vertical_field_T=self.magnetic.vertical_field(t_s),
            effective_charge=effective_charge,
        )
        tau_e, tau_i, tau_energy = loss_times.electron_s, loss_times.ion_s, loss_times.energy_s
        S_ion_bulk = ne * nn * rates.K_ion_m3_s
        hot_rate_fn = getattr(self.rates_model, "hot_ionization_rate", None)
        K_ion_hot = (
            float(hot_rate_fn(state.hot_mean_energy_eV, EoverN_Td))
            if callable(hot_rate_fn)
            else 0.0
        )
        S_ion_hot = max(state.n_hot_m3, 0.0) * nn * K_ion_hot
        S_ion = S_ion_bulk + S_ion_hot
        S_rec = rates.alpha_rec_m3_s * ne * ni
        S_att = ne * nn * rates.K_att_m3_s
        S_e_wall = ne / tau_e
        S_i_wall = ni / tau_i
        S_secondary = cfg.gas.secondary_emission * S_i_wall
        S_seed = cfg.initial.seed_rate_m3_s

        kinetic_terms = None
        if self.kinetic_closure is not None:
            kinetic_terms = self.kinetic_closure.evaluate(t_s)
            # The kinetic rates replace the corresponding volume-averaged 0D
            # closures.  Adding them would double count ionization and wall
            # losses already resolved in f(s,u_parallel,u_perp,t).
            S_ion = max(float(kinetic_terms.ionization_rate_m3_s), 0.0)
            S_ion_bulk = S_ion
            S_ion_hot = 0.0
            S_e_wall = max(float(kinetic_terms.electron_wall_loss_rate_m3_s), 0.0)

        # 功率密度（W/m^3）。线辐射/分子通道在高保真 provider 中替换。
        P_ohm = circuit.conductivity_S_m * circuit.E_loop_V_m**2
        P_rf_total = (
            rf_terms.power_density_W_m3
            if kinetic_terms is None
            else max(float(kinetic_terms.absorbed_rf_power_W), 0.0) / self.volume_m3
        )
        P_rf_bulk = (
            P_rf_total
            if kinetic_terms is not None
            else P_rf_total - rf_terms.hot_power_density_W_m3
        )
        P_ion = S_ion * cfg.gas.ionization_energy_eV * E_CHARGE_C
        P_exc = ne * nn * rates.K_exc_m3_s * cfg.gas.excitation_energy_eV * E_CHARGE_C
        P_rad = 5.35e-37 * ne * ne * math.sqrt(max(state.Te_eV, 1.0e-3))
        P_wall = 2.5 * state.Te_eV * E_CHARGE_C * S_i_wall
        mean_bulk_energy = state.electron_energy_J_m3 / max(ne, 1.0)
        P_particle = mean_bulk_energy * (S_rec + S_att + S_e_wall)
        P_loss = P_ion + P_exc + P_rad + P_wall + P_particle + state.electron_energy_J_m3 / tau_energy
        P_balance = P_ohm + P_rf_bulk - P_loss

        nu_ion = nn * rates.K_ion_m3_s
        nu_att = nn * rates.K_att_m3_s
        nu_rec = ni * rates.alpha_rec_m3_s
        nu_loss = 1.0 / tau_e
        net_growth = nu_ion + S_secondary / max(ne, 1.0) - nu_att - nu_rec - nu_loss
        vte = self.rates_model.electron_thermal_speed_m_s(state.Te_eV)
        alpha_eff = (nu_ion - nu_att) / max(vte, 1.0) - 1.0 / max(topology.connection_length_m, 1.0e-9)
        # Dreicer 比仅作预警指标；完整 runaway 需动理学方程。
        E_dreicer = M_E_KG * max(rates.nu_ei_s, 1.0) * vte / E_CHARGE_C
        omega_ce = E_CHARGE_C * cfg.geometry.B_t_T / M_E_KG
        nu_total = max(circuit.nu_en_s + circuit.nu_ei_s, 1.0)
        lambda_D = math.sqrt(EPS0_F_M * max(state.Te_eV, 1.0e-3) * E_CHARGE_C / max(ne * E_CHARGE_C**2, 1.0e-40))

        primary_beam = rf_terms.components[0] if rf_terms.components else None
        launch_R = primary_beam.launch_R_m if primary_beam is not None else (
            cfg.rf.launch.R_m if cfg.rf.launch.R_m is not None else cfg.geometry.R0_m
        )
        launch_Z = primary_beam.launch_Z_m if primary_beam is not None else cfg.rf.launch.Z_m
        launch_point = self.magnetic.evaluate(launch_R, launch_Z, state.Ip_A, t_s)
        neutral_pressure = nn * K_B_J_K * cfg.gas.Tn_K
        target_pressure = cfg.gas.pressure_at(t_s)
        values: Dict[str, float | str] = {
            "ne_m3": ne,
            "ni_m3": ni,
            "nn_m3": nn,
            "electron_energy_J_m3": max(state.electron_energy_J_m3, 0.0),
            "hot_energy_J_m3": max(state.hot_energy_J_m3, 0.0),
            "Te_eV": state.Te_eV,
            "Ti_eV": cfg.initial.Ti_eV,
            "Ip_A": state.Ip_A,
            "n_hot_m3": max(state.n_hot_m3, 0.0),
            "hot_mean_energy_eV": state.hot_mean_energy_eV,
            "E_loop_V_m": circuit.E_loop_V_m,
            "loop_voltage_V": circuit.loop_voltage_V,
            "effective_loop_voltage_V": circuit.effective_loop_voltage_V,
            "cs_mode": cfg.circuit.cs_mode,
            "Z_eff_used": circuit.effective_charge,
            "EoverN_Td": EoverN_Td,
            "neutral_pressure_Pa": neutral_pressure,
            "target_pressure_Pa": target_pressure,
            "pL_Pa_m": neutral_pressure * topology.connection_length_m,
            "B_vertical_T": self.magnetic.vertical_field(t_s),
            "field_configuration": self.magnetic.configuration,
            "B_R_T": launch_point.B_R_T,
            "B_Z_T": launch_point.B_Z_T,
            "B_phi_T": launch_point.B_phi_T,
            "B_magnitude_T": launch_point.magnitude_T,
            "mirror_ratio": magnetic_state.mirror_ratio,
            "trapped_fraction": magnetic_state.trapped_fraction,
            "decay_index": magnetic_state.decay_index,
            "vertical_alignment": magnetic_state.vertical_alignment,
            "connection_length_factor": magnetic_state.connection_length_factor,
            "field_loss_multiplier": magnetic_state.loss_multiplier,
            "mirror_confinement_factor": magnetic_state.mirror_confinement_factor,
            "tail_confinement_metric": magnetic_state.tail_confinement_metric,
            "loss_configuration_multiplier": getattr(loss_times, "configuration_multiplier", 1.0),
            "K_ion_m3_s": rates.K_ion_m3_s,
            "K_exc_m3_s": rates.K_exc_m3_s,
            "nu_ion_s": nu_ion,
            "nu_loss_s": nu_loss,
            "nu_en_s": circuit.nu_en_s,
            "nu_ei_s": circuit.nu_ei_s,
            "net_growth_s": net_growth,
            "townsend_alpha_eff_m": alpha_eff,
            "Dreicer_ratio": abs(E_loop) / max(E_dreicer, 1.0e-30),
            "omega_ce_over_nu": omega_ce / nu_total,
            "lambda_D_over_a": lambda_D / cfg.geometry.minor_radius_m,
            "mean_free_path_over_a": (vte / nu_total) / cfg.geometry.minor_radius_m,
            "ionization_fraction": ne / max(ne + nn, 1.0),
            "P_ohm_W_m3": P_ohm,
            "P_rf_W_m3": P_rf_total,
            "P_rf_bulk_W_m3": P_rf_bulk,
            "P_ion_W_m3": P_ion,
            "P_exc_W_m3": P_exc,
            "P_rad_W_m3": P_rad,
            "P_wall_W_m3": P_wall,
            "P_particle_W_m3": P_particle,
            "P_loss_W_m3": P_loss,
            "P_balance_W_m3": P_balance,
            "sigma_S_m": circuit.conductivity_S_m,
            "R_plasma_ohm": circuit.plasma_resistance_ohm,
            "I_ohmic_A": circuit.I_ohmic_A,
            "I_bulk_A": circuit.I_ohmic_A,
            "I_rf_A": circuit.I_rf_A,
            "I_kinetic_A": circuit.I_kinetic_A,
            "I_orbit_A": circuit.I_orbit_A,
            "I_tail_A": circuit.I_tail_A,
            "I_tpc_A": circuit.I_tpc_A,
            "I_vfc_A": circuit.I_vfc_A,
            "I_vtf_A": circuit.I_vtf_A,
            "I_vertical_charge_A": circuit.I_vertical_charge_A,
            "I_drive_A": circuit.I_drive_A,
            "current_drive_model": circuit.current_drive_model,
            "I_component_sum_A": circuit.I_ohmic_A + circuit.I_drive_A,
            "dIp_dt_A_s": circuit.dIp_dt_A_s,
            "rf_power_absorbed_W": rf_terms.power_absorbed_W,
            "rf_accessibility": rf_terms.accessibility,
            "omega_ce_over_omega_rf": rf_terms.omega_ce_over_omega_rf,
            "closed_fraction": topology.closed_fraction,
            "candidate_radius_m": topology.candidate_radius_m,
            "Bp_edge_T": topology.Bp_edge_T,
            "connection_length_effective_m": topology.connection_length_m,
            "topology_confidence": topology.confidence,
            "S_ion_m3_s": S_ion,
            "S_ion_bulk_m3_s": S_ion_bulk,
            "S_ion_hot_m3_s": S_ion_hot,
            "K_ion_hot_m3_s": K_ion_hot,
            "S_rec_m3_s": S_rec,
            "S_att_m3_s": S_att,
            "S_e_wall_m3_s": S_e_wall,
            "S_i_wall_m3_s": S_i_wall,
            "S_secondary_m3_s": S_secondary,
            "S_seed_m3_s": S_seed,
            "kinetic_particle_power_closure": (
                "disabled" if kinetic_terms is None else "open_field_line_1d2v_replacement"
            ),
        }
        if kinetic_terms is not None:
            values.update(
                {
                    "kinetic_ne_volume_average_m3": float(kinetic_terms.electron_density_m3),
                    "kinetic_ni_volume_average_m3": float(kinetic_terms.ion_density_m3),
                    "kinetic_nn_volume_average_m3": float(kinetic_terms.neutral_density_m3),
                    "kinetic_absorbed_rf_power_W": float(kinetic_terms.absorbed_rf_power_W),
                }
            )
        values.update(rf_terms.diagnostics())
        mechanism_values = {
            "ohmic": abs(circuit.I_ohmic_A),
            "direct_rf": abs(circuit.I_rf_A),
            "hot_tail": abs(circuit.I_tail_A),
            "vertical_charge": abs(circuit.I_vertical_charge_A),
        }
        mechanism_total = sum(mechanism_values.values())
        values["dominant_current_mechanism"] = (
            max(mechanism_values, key=mechanism_values.get) if mechanism_total > 0.0 else "none"
        )
        for name, magnitude in mechanism_values.items():
            values[f"current_fraction_{name}"] = magnitude / max(mechanism_total, 1.0e-300)
        for index, beam in enumerate(rf_terms.components):
            beam_point = self.magnetic.evaluate(beam.launch_R_m, beam.launch_Z_m, state.Ip_A, t_s)
            local_omega_ce = E_CHARGE_C * beam_point.magnitude_T / M_E_KG
            values[f"rf_{index}_B_launch_T"] = beam_point.magnitude_T
            values[f"rf_{index}_omega_ce_over_omega_rf_local"] = local_omega_ce / (
                2.0 * math.pi * beam.frequency_Hz
            )
        values.update({k: v for k, v in self.magnetic.diagnostics(state.Ip_A, t_s).items() if k != "field_configuration"})
        species_terms = None
        if self.species_wall_model is not None:
            if extended_state is None:
                raise TypeError("启用 species_wall 后 evaluate 必须接收 ExtendedGlobalState")
            species_terms = self.species_wall_model.evaluate(
                extended_state.species_wall,
                state,
                values,
                cfg.initial.Ti_eV,
            )
            extra_atomic_power = species_terms.radiation_power_W_m3 + species_terms.ionization_power_W_m3
            values["P_rad_base_W_m3"] = values["P_rad_W_m3"]
            values["P_rad_W_m3"] = float(values["P_rad_W_m3"]) + species_terms.radiation_power_W_m3
            values["P_loss_W_m3"] = float(values["P_loss_W_m3"]) + extra_atomic_power
            values["P_balance_W_m3"] = float(values["P_balance_W_m3"]) - extra_atomic_power
            values.update(species_terms.diagnostics)

        impurity_charge_wall_loss = (
            species_terms.positive_charge_wall_loss_m3_s if species_terms is not None else 0.0
        )
        charge_terms = self.charge_closure_model.evaluate(
            Te_eV=state.Te_eV,
            free_electron_wall_loss_m3_s=S_e_wall,
            positive_charge_wall_loss_m3_s=S_i_wall + impurity_charge_wall_loss,
            secondary_electron_source_m3_s=S_secondary,
            attachment_loss_m3_s=S_att,
        )
        effective_electron_loss = charge_terms.electron_wall_loss_m3_s
        particle_power_delta = mean_bulk_energy * (effective_electron_loss - S_e_wall)
        values["S_e_wall_m3_s"] = effective_electron_loss
        values["nu_loss_s"] = effective_electron_loss / max(ne, 1.0)
        values["net_growth_s"] = (
            nu_ion
            + S_secondary / max(ne, 1.0)
            - nu_att
            - nu_rec
            - effective_electron_loss / max(ne, 1.0)
        )
        values["P_particle_W_m3"] = float(values["P_particle_W_m3"]) + particle_power_delta
        values["P_loss_W_m3"] = float(values["P_loss_W_m3"]) + particle_power_delta
        values["P_balance_W_m3"] = float(values["P_balance_W_m3"]) - particle_power_delta
        values.update(charge_terms.diagnostics())
        values.setdefault("electron_density_quasineutral_m3", ni)
        values.setdefault("charge_balance_residual_m3", ne - ni)
        values["charge_balance_relative"] = float(values["charge_balance_residual_m3"]) / max(
            ne, float(values["electron_density_quasineutral_m3"]), 1.0
        )
        return PhysicsEvaluation(rates, rf_terms, circuit, topology, values, species_terms, charge_terms)

    def rhs(self, t_s: float, vector: np.ndarray) -> np.ndarray:
        """供 scipy.solve_ivp 使用的右端项。"""

        model_state = self.state_from_vector(vector).projected()
        state = model_state.core if isinstance(model_state, ExtendedGlobalState) else model_state
        ev = self.evaluate(t_s, model_state)
        v = ev.values
        cfg = self.config
        nn = max(state.nn_m3, 0.0)
        target_nn = cfg.gas.pressure_at(t_s) / (K_B_J_K * cfg.gas.Tn_K)
        S_ion = v["S_ion_m3_s"]
        S_rec = v["S_rec_m3_s"]
        S_att = v["S_att_m3_s"]
        S_e_wall = v["S_e_wall_m3_s"]
        S_i_wall = v["S_i_wall_m3_s"]

        dne = S_ion + v["S_secondary_m3_s"] - S_rec - S_att - S_e_wall + cfg.initial.seed_rate_m3_s
        if ev.species_wall is not None:
            dne += ev.species_wall.electron_source_m3_s
        dni = S_ion - S_rec - S_i_wall + cfg.initial.seed_rate_m3_s
        feed = target_nn / cfg.gas.pump_time_s
        pump = nn / cfg.gas.pump_time_s
        recycle = cfg.gas.recycling_coefficient * S_i_wall
        if ev.species_wall is not None and self._has_fuel_species:
            recycle = ev.species_wall.fuel_neutral_source_m3_s
        dnn = -S_ion - cfg.initial.seed_rate_m3_s + S_rec + recycle - pump + feed

        due = v["P_balance_W_m3"]
        # 新产生/损失的粒子携带的冷能量在 P_particle 中已近似处理。
        kinetic_replaces_hot = self.kinetic_closure is not None
        dnh = (
            (0.0 if kinetic_replaces_hot else ev.rf.hot_particle_source_m3_s)
            - max(state.n_hot_m3, 0.0) / cfg.rf.hot_loss_time_s
            - v["S_ion_hot_m3_s"]
        )
        duh = (
            (0.0 if kinetic_replaces_hot else ev.rf.hot_power_density_W_m3)
            - max(state.hot_energy_J_m3, 0.0) / cfg.rf.hot_loss_time_s
            - max(state.hot_energy_J_m3, 0.0) / cfg.rf.hot_scattering_time_s
            - v["S_ion_hot_m3_s"] * cfg.gas.ionization_energy_eV * E_CHARGE_C
        )
        core_derivative = np.asarray([dne, dni, dnn, due, ev.circuit.dIp_dt_A_s, dnh, duh], dtype=float)
        if ev.species_wall is None:
            return core_derivative
        return np.concatenate((core_derivative, ev.species_wall.derivative))
