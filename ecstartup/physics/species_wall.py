"""Independent DYON-like atomic species and wall-inventory model.

This module contains no DYON source code.  It follows the same global-model
decomposition: adjacent charge-state balances, wall retention/recycling,
sputtering, and species-resolved radiation.  The built-in atomic rates are
analytic surrogates; production work should inject an ADAS-backed provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Protocol, Sequence

import numpy as np

from ..config import AtomicSpeciesConfig, DYONSpeciesWallConfig, GeometryConfig
from ..constants import E_CHARGE_C
from ..state import GlobalState


class AtomicRateProvider(Protocol):
    """Replaceable interface for SCD/ACD/PLT/PRB-like atomic data."""

    label: str

    def ionization_rate_m3_s(self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19) -> float: ...

    def recombination_rate_m3_s(self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19) -> float: ...

    def radiation_coefficient_W_m3(
        self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19
    ) -> float: ...

    def ionization_energy_eV(self, species: AtomicSpeciesConfig, charge: int) -> float: ...


class AnalyticAtomicRateProvider:
    """Positive, thresholded placeholder rates, explicitly not ADAS data."""

    label = "analytic_surrogate"

    @staticmethod
    def _te(Te_eV: float) -> float:
        return max(float(Te_eV), 1.0e-3)

    def ionization_energy_eV(self, species: AtomicSpeciesConfig, charge: int) -> float:
        if not 0 <= charge < species.max_charge:
            raise ValueError("ionization charge 必须满足 0 <= q < max_charge")
        if species.ionization_energies_eV is not None:
            return float(species.ionization_energies_eV[charge])
        # Hydrogenic scaling is only a smooth fallback, not an atomic database.
        return 13.6 * (charge + 1) ** 2

    def ionization_rate_m3_s(self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19) -> float:
        Te = self._te(Te_eV)
        threshold = self.ionization_energy_eV(species, charge)
        return max(
            0.0,
            float(species.ionization_rate_scale_m3_s)
            * math.sqrt(Te / max(threshold, 1.0e-6))
            * math.exp(-threshold / Te)
            / (charge + 1),
        )

    def recombination_rate_m3_s(self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19) -> float:
        if not 1 <= charge <= species.max_charge:
            raise ValueError("recombination charge 必须满足 1 <= q <= max_charge")
        Te = self._te(Te_eV)
        return max(
            0.0,
            float(species.recombination_rate_scale_m3_s)
            * charge**2
            * (Te / 1.0) ** -0.7,
        )

    def radiation_coefficient_W_m3(
        self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19
    ) -> float:
        Te = self._te(Te_eV)
        # A bounded shape keeps the surrogate numerically useful while making
        # the dimensional coefficient and its calibration point explicit.
        temperature_shape = math.sqrt(Te / (Te + 10.0))
        charge_shape = 1.0 + float(charge)
        return max(0.0, float(species.radiation_coefficient_W_m3) * charge_shape * temperature_shape)


@dataclass(frozen=True)
class SpeciesWallLayout:
    """Deterministic mapping between named states and an ODE vector."""

    species: tuple[AtomicSpeciesConfig, ...]
    walls: tuple
    density_slices: Mapping[str, slice]
    wall_indices: Mapping[str, int]
    vector_names: tuple[str, ...]

    @classmethod
    def from_config(cls, config: DYONSpeciesWallConfig) -> "SpeciesWallLayout":
        species = tuple(sorted(config.species, key=lambda item: item.symbol))
        walls = tuple(sorted(config.wall_materials, key=lambda item: item.name))
        names: list[str] = []
        density_slices: Dict[str, slice] = {}
        cursor = 0
        for item in species:
            if item.role == "fuel":
                continue
            density_slices[item.symbol] = slice(cursor, cursor + item.max_charge + 1)
            names.extend(f"n_{item.symbol}{charge}_m3" for charge in range(item.max_charge + 1))
            cursor += item.max_charge + 1
        wall_indices: Dict[str, int] = {}
        for wall in walls:
            wall_indices[wall.name] = cursor
            names.append(f"wall_inventory_{wall.species}_m2")
            cursor += 1
        return cls(species, walls, density_slices, wall_indices, tuple(names))

    @property
    def size(self) -> int:
        return len(self.vector_names)

    def initial_state(self) -> "SpeciesWallState":
        vector = np.zeros(self.size, dtype=float)
        by_symbol = {item.symbol: item for item in self.species}
        for symbol, section in self.density_slices.items():
            vector[section] = np.asarray(by_symbol[symbol].initial_charge_densities_m3, dtype=float)
        wall_by_name = {item.name: item for item in self.walls}
        for name, index in self.wall_indices.items():
            vector[index] = float(wall_by_name[name].initial_inventory_m2)
        return SpeciesWallState(self, vector)


@dataclass
class SpeciesWallState:
    layout: SpeciesWallLayout
    vector: np.ndarray

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float).copy()
        if self.vector.shape != (self.layout.size,):
            raise ValueError(f"species/wall 状态长度必须为 {self.layout.size}")

    @classmethod
    def from_vector(cls, layout: SpeciesWallLayout, vector: Iterable[float]) -> "SpeciesWallState":
        return cls(layout, np.asarray(list(vector), dtype=float))

    def to_vector(self) -> np.ndarray:
        return self.vector.copy()

    def projected(self) -> "SpeciesWallState":
        return SpeciesWallState(self.layout, np.maximum(self.vector, 0.0))

    def charge_densities(self, symbol: str) -> np.ndarray:
        try:
            return self.vector[self.layout.density_slices[symbol]].copy()
        except KeyError as exc:
            raise KeyError(f"{symbol} 不是显式积分的 impurity") from exc

    def wall_inventory_m2(self, wall_name: str) -> float:
        return float(self.vector[self.layout.wall_indices[wall_name]])


@dataclass
class ExtendedGlobalState:
    """Legacy seven-state core plus the optional variable-size sidecar."""

    core: GlobalState
    species_wall: SpeciesWallState

    def to_vector(self) -> np.ndarray:
        return np.concatenate((self.core.to_vector(), self.species_wall.to_vector()))

    @classmethod
    def from_vector(cls, layout: SpeciesWallLayout, vector: Sequence[float]) -> "ExtendedGlobalState":
        values = np.asarray(vector, dtype=float)
        if values.size != 7 + layout.size:
            raise ValueError(f"扩展状态长度必须为 {7 + layout.size}")
        return cls(GlobalState.from_vector(values[:7]), SpeciesWallState(layout, values[7:]))

    def projected(self) -> "ExtendedGlobalState":
        return ExtendedGlobalState(self.core.projected(), self.species_wall.projected())

    def __getattr__(self, name: str):
        # Preserve read-only consumers such as comparison.py and plotting code.
        return getattr(self.core, name)


@dataclass(frozen=True)
class SpeciesWallTerms:
    derivative: np.ndarray
    electron_source_m3_s: float
    fuel_neutral_source_m3_s: float
    radiation_power_W_m3: float
    ionization_power_W_m3: float
    positive_charge_wall_loss_m3_s: float
    diagnostics: Dict[str, float | str]


class DYONSpeciesWallModel:
    """Adjacent-charge atomic balances coupled to a conservative wall model."""

    def __init__(
        self,
        config: DYONSpeciesWallConfig,
        geometry: GeometryConfig,
        atomic_rates: AtomicRateProvider | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("DYONSpeciesWallModel 只能由 enabled=true 的配置构造")
        self.config = config
        self.geometry = geometry
        self.layout = SpeciesWallLayout.from_config(config)
        self.atomic_rates = atomic_rates if atomic_rates is not None else AnalyticAtomicRateProvider()
        self.volume_m3 = float(geometry.resolved_volume_m3)
        self.wall_area_m2 = float(geometry.resolved_wall_area_m2)
        self._species = {item.symbol: item for item in self.layout.species}

    def initial_state(self) -> SpeciesWallState:
        return self.layout.initial_state()

    def effective_charge(self, state: SpeciesWallState, core: GlobalState) -> float:
        """Return sum(n_s Z_s^2)/sum(n_s Z_s) directly from charge states."""

        state = state.projected()
        charge_density = max(float(core.ni_m3), 0.0)
        charge_squared_density = charge_density
        for species in self.layout.species:
            if species.role == "fuel":
                continue
            densities = state.charge_densities(species.symbol)
            charges = np.arange(species.max_charge + 1, dtype=float)
            charge_density += float(np.dot(charges, densities))
            charge_squared_density += float(np.dot(charges**2, densities))
        return max(1.0, charge_squared_density / max(charge_density, 1.0))

    def evaluate(
        self,
        state: SpeciesWallState,
        core: GlobalState,
        core_values: Mapping[str, float | str],
        Ti_eV: float,
    ) -> SpeciesWallTerms:
        state = state.projected()
        ne = max(float(core.ne_m3), 0.0)
        Te = max(float(core.Te_eV), 1.0e-3)
        derivative = np.zeros(self.layout.size, dtype=float)
        diagnostics: Dict[str, float | str] = {"atomic_rate_backend": self.atomic_rates.label}
        electron_source = 0.0
        radiation_total = 0.0
        ionization_power = 0.0
        charge_density = max(float(core.ni_m3), 0.0)
        charge_squared_density = max(float(core.ni_m3), 0.0)
        wall_loss_by_species: Dict[str, float] = {symbol: 0.0 for symbol in self._species}
        neutral_pump_total = 0.0
        impurity_charge_wall_loss = 0.0

        fuel = next((item for item in self.layout.species if item.role == "fuel"), None)
        if fuel is not None:
            fuel_densities = (max(float(core.nn_m3), 0.0), max(float(core.ni_m3), 0.0))
            for charge, density in enumerate(fuel_densities):
                diagnostics[f"n_{fuel.symbol}{charge}_m3"] = density
                power = ne * density * self.atomic_rates.radiation_coefficient_W_m3(fuel, charge, Te, ne)
                diagnostics[f"P_rad_{fuel.symbol}{charge}_W_m3"] = power
                radiation_total += power
            wall_loss_by_species[fuel.symbol] = max(float(core_values["S_i_wall_m3_s"]), 0.0)

        inv_tau_i = max(float(core_values["S_i_wall_m3_s"]), 0.0) / max(float(core.ni_m3), 1.0)
        for species in self.layout.species:
            if species.role == "fuel":
                continue
            section = self.layout.density_slices[species.symbol]
            densities = state.vector[section]
            local = np.zeros_like(densities)
            species_radiation = 0.0
            species_ionization_power = 0.0

            for charge, density in enumerate(densities):
                diagnostics[f"n_{species.symbol}{charge}_m3"] = float(density)
                rad = ne * density * self.atomic_rates.radiation_coefficient_W_m3(species, charge, Te, ne)
                diagnostics[f"P_rad_{species.symbol}{charge}_W_m3"] = rad
                species_radiation += rad
                charge_density += charge * density
                charge_squared_density += charge * charge * density

            for charge in range(species.max_charge):
                rate = ne * densities[charge] * self.atomic_rates.ionization_rate_m3_s(species, charge, Te, ne)
                local[charge] -= rate
                local[charge + 1] += rate
                electron_source += rate
                energy = self.atomic_rates.ionization_energy_eV(species, charge)
                species_ionization_power += rate * energy * E_CHARGE_C
                diagnostics[f"S_ion_{species.symbol}{charge}_m3_s"] = rate

            for charge in range(1, species.max_charge + 1):
                rate = ne * densities[charge] * self.atomic_rates.recombination_rate_m3_s(species, charge, Te, ne)
                local[charge] -= rate
                local[charge - 1] += rate
                electron_source -= rate
                diagnostics[f"S_rec_{species.symbol}{charge}_m3_s"] = rate

            charged_wall_loss = 0.0
            for charge in range(1, species.max_charge + 1):
                loss = densities[charge] * inv_tau_i * species.wall_loss_multiplier
                local[charge] -= loss
                charged_wall_loss += loss
                impurity_charge_wall_loss += charge * loss
            wall_loss_by_species[species.symbol] = charged_wall_loss

            if species.neutral_pump_time_s is not None:
                neutral_pump = densities[0] / species.neutral_pump_time_s
                local[0] -= neutral_pump
                neutral_pump_total += neutral_pump
                diagnostics[f"S_pump_{species.symbol}_m3_s"] = neutral_pump

            derivative[section] = local
            diagnostics[f"S_wall_{species.symbol}_m3_s"] = charged_wall_loss
            diagnostics[f"P_rad_{species.symbol}_W_m3"] = species_radiation
            diagnostics[f"P_ion_{species.symbol}_W_m3"] = species_ionization_power
            radiation_total += species_radiation
            ionization_power += species_ionization_power

        total_incident = sum(wall_loss_by_species.values())
        fuel_neutral_source = 0.0
        sputter_total = 0.0
        wall_equivalent_derivative = 0.0
        configured_wall_species = {wall.species for wall in self.layout.walls}
        untracked_wall_sink = sum(
            loss for symbol, loss in wall_loss_by_species.items() if symbol not in configured_wall_species
        )
        max_incident_energy_eV = 0.0
        for wall in self.layout.walls:
            species = self._species[wall.species]
            incoming = wall_loss_by_species[wall.species]
            inventory = state.wall_inventory_m2(wall.name)
            desorption_surface = inventory / wall.desorption_time_s if wall.desorption_time_s else 0.0
            desorption_volume = desorption_surface * self.wall_area_m2 / self.volume_m3
            retained_surface = (1.0 - wall.recycling_coefficient) * incoming * self.volume_m3 / self.wall_area_m2
            derivative[self.layout.wall_indices[wall.name]] = retained_surface - desorption_surface

            wall_incident_energy = max(float(Ti_eV), 0.0) + wall.incident_energy_factor * Te
            max_incident_energy_eV = max(max_incident_energy_eV, wall_incident_energy)
            if wall_incident_energy > float(wall.physical_sputtering_threshold_eV):
                threshold_factor = 1.0 - float(wall.physical_sputtering_threshold_eV) / wall_incident_energy
            else:
                threshold_factor = 0.0
            effective_yield = wall.physical_sputtering_yield * threshold_factor + wall.chemical_sputtering_yield
            sputter_source = total_incident * effective_yield
            returned = wall.recycling_coefficient * incoming + desorption_volume + sputter_source
            if species.role == "fuel":
                fuel_neutral_source += returned
            else:
                derivative[self.layout.density_slices[species.symbol].start] += returned

            sputter_total += sputter_source
            wall_equivalent_derivative += derivative[self.layout.wall_indices[wall.name]] * self.wall_area_m2 / self.volume_m3
            diagnostics[f"wall_inventory_{wall.species}_m2"] = inventory
            diagnostics[f"S_recycle_{wall.species}_m3_s"] = wall.recycling_coefficient * incoming + desorption_volume
            diagnostics[f"S_sputter_{wall.species}_m3_s"] = sputter_source
            diagnostics[f"sputtering_yield_{wall.species}"] = effective_yield
            diagnostics[f"wall_incident_energy_{wall.species}_eV"] = wall_incident_energy

        impurity_plasma_derivative = 0.0
        for species in self.layout.species:
            if species.role == "impurity":
                impurity_plasma_derivative += float(np.sum(derivative[self.layout.density_slices[species.symbol]]))
        fuel_balance_derivative = -wall_loss_by_species.get(fuel.symbol, 0.0) + fuel_neutral_source if fuel else 0.0
        nuclei_residual = (
            impurity_plasma_derivative
            + fuel_balance_derivative
            + wall_equivalent_derivative
            + neutral_pump_total
            + untracked_wall_sink
            - sputter_total
        )
        balance_scale = max(
            1.0,
            total_incident
            + fuel_neutral_source
            + neutral_pump_total
            + sputter_total
            + abs(wall_equivalent_derivative),
        )
        diagnostics.update(
            {
                "electron_density_quasineutral_m3": charge_density,
                "charge_balance_residual_m3": ne - charge_density,
                "Z_eff": charge_squared_density / max(charge_density, 1.0),
                "P_rad_species_W_m3": radiation_total,
                "P_ion_species_W_m3": ionization_power,
                "S_e_species_m3_s": electron_source,
                "S_impurity_charge_wall_m3_s": impurity_charge_wall_loss,
                "S_fuel_recycle_m3_s": fuel_neutral_source,
                "species_particle_balance_residual_m3_s": nuclei_residual,
                "species_particle_balance_relative": nuclei_residual / balance_scale,
                "wall_incident_energy_eV": max_incident_energy_eV,
            }
        )
        return SpeciesWallTerms(
            derivative=derivative,
            electron_source_m3_s=electron_source,
            fuel_neutral_source_m3_s=fuel_neutral_source,
            radiation_power_W_m3=radiation_total,
            ionization_power_W_m3=ionization_power,
            positive_charge_wall_loss_m3_s=impurity_charge_wall_loss,
            diagnostics=diagnostics,
        )
