"""Versioned atomic/molecular data adapters and EEDF-consistent rates.

Raw LXCat, AMJUEL/HYDHEL and ADAS distributions use different formats and
licenses.  linEC therefore consumes a normalized, checksummed snapshot and
records its provenance.  The adapters do not ship or fabricate database
values; production runs must provide the actual downloaded tables.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from ..constants import E_CHARGE_C, M_E_KG
from ..config import GasConfig
from .rates import CollisionRateModel, RateCoefficients


@dataclass(frozen=True)
class AtomicDataProvenance:
    database: str
    version: str
    retrieved_date: str
    species: str
    source_url: str
    citation: str
    license_note: str


@dataclass(frozen=True)
class CrossSectionTable:
    process: str
    energy_eV: np.ndarray
    cross_section_m2: np.ndarray
    provenance: AtomicDataProvenance

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy_eV, dtype=float)
        cross_section = np.asarray(self.cross_section_m2, dtype=float)
        if energy.ndim != 1 or cross_section.shape != energy.shape or len(energy) < 2:
            raise ValueError("cross-section energy/value arrays must be equal-length 1-D arrays")
        if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(cross_section)):
            raise ValueError("cross-section table contains non-finite values")
        if energy[0] < 0.0 or np.any(np.diff(energy) <= 0.0) or np.any(cross_section < 0.0):
            raise ValueError("cross-section energies must increase and sigma must be non-negative")
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "cross_section_m2", cross_section)

    def sigma_m2(self, energy_eV: np.ndarray) -> np.ndarray:
        energy = np.asarray(energy_eV, dtype=float)
        return np.interp(energy, self.energy_eV, self.cross_section_m2, left=0.0, right=0.0)

    def maxwellian_rate_m3_s(self, temperature_eV: float) -> float:
        temperature = float(temperature_eV)
        if temperature <= 0.0:
            raise ValueError("temperature_eV must be positive")
        energy = self.energy_eV
        probability_per_eV = (
            2.0 / math.sqrt(math.pi)
            * np.sqrt(energy)
            / temperature**1.5
            * np.exp(-energy / temperature)
        )
        speed = np.sqrt(2.0 * E_CHARGE_C * energy / M_E_KG)
        return float(np.trapz(self.cross_section_m2 * speed * probability_per_eV, energy))

    def eedf_rate_m3_s(self, grid, distribution: np.ndarray) -> float:
        values = np.asarray(distribution, dtype=float)
        if values.shape != grid.shape or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("EEDF must be finite, non-negative and match its kinetic grid")
        v_parallel, v_perp = grid.mesh()
        speed = np.sqrt(v_parallel**2 + v_perp**2)
        gamma = 1.0 / np.sqrt(np.maximum(1.0 - np.minimum(speed / 299_792_458.0, 1.0 - 1.0e-14) ** 2, 1.0e-14))
        energy_eV = (gamma - 1.0) * M_E_KG * 299_792_458.0**2 / E_CHARGE_C
        weights = grid.phase_volume_weights_m3_s3
        density = float(np.sum(values * weights))
        if density <= 0.0:
            return 0.0
        return float(np.sum(self.sigma_m2(energy_eV) * speed * values * weights) / density)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_normalized_cross_sections(manifest_path: str | Path) -> dict[str, CrossSectionTable]:
    """Load a normalized LXCat/AMJUEL/HYDHEL snapshot after checksum checks."""

    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    provenance = AtomicDataProvenance(
        database=str(payload["database"]),
        version=str(payload["version"]),
        retrieved_date=str(payload["retrieved_date"]),
        species=str(payload["species"]),
        source_url=str(payload["source_url"]),
        citation=str(payload["citation"]),
        license_note=str(payload["license_note"]),
    )
    tables: dict[str, CrossSectionTable] = {}
    for item in payload.get("processes", []):
        process = str(item["process"])
        table_path = manifest_file.parent / str(item["file"])
        expected_hash = str(item["sha256"]).lower()
        actual_hash = _sha256(table_path)
        if actual_hash != expected_hash:
            raise ValueError(f"checksum mismatch for atomic process {process}")
        energy = []
        sigma = []
        with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"energy_eV", "cross_section_m2"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"normalized cross-section file {table_path} lacks {sorted(required)}")
            for row in reader:
                energy.append(float(row["energy_eV"]))
                sigma.append(float(row["cross_section_m2"]))
        tables[process] = CrossSectionTable(process, np.asarray(energy), np.asarray(sigma), provenance)
    if not tables:
        raise ValueError("atomic-data manifest contains no processes")
    return tables


class CrossSectionRateModel(CollisionRateModel):
    """RateProvider backed by versioned cross sections rather than fits."""

    def __init__(self, gas: GasConfig, tables: Mapping[str, CrossSectionTable]) -> None:
        super().__init__(gas)
        self.tables = dict(tables)
        missing = {"ionization", "excitation", "momentum"} - set(self.tables)
        if missing:
            raise ValueError(f"cross-section dataset lacks required processes: {sorted(missing)}")

    def _maxwellian(self, process: str, temperature_eV: float) -> float:
        table = self.tables.get(process)
        return 0.0 if table is None else table.maxwellian_rate_m3_s(temperature_eV)

    def eedf_rates(self, grid, distribution: np.ndarray) -> dict[str, float]:
        return {
            process: table.eedf_rate_m3_s(grid, distribution)
            for process, table in self.tables.items()
        }

    def evaluate(self, Te_eV: float, ne_m3: float, nn_m3: float, EoverN_Td: float) -> RateCoefficients:
        coulomb = super().evaluate(Te_eV, ne_m3, nn_m3, EoverN_Td)
        return RateCoefficients(
            self._maxwellian("ionization", Te_eV),
            self._maxwellian("excitation", Te_eV),
            self._maxwellian("attachment", Te_eV),
            self.gas.recombination_coeff_m3_s,
            self._maxwellian("momentum", Te_eV),
            coulomb.nu_ei_s,
            next(iter(self.tables.values())).provenance.database,
        )


__all__ = [
    "AtomicDataProvenance",
    "CrossSectionTable",
    "CrossSectionRateModel",
    "load_normalized_cross_sections",
]
