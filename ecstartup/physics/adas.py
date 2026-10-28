"""Native OpenADAS ADF11 reader and species-wall rate provider.

ADF11 stores base-10 logarithms on electron-density (cm^-3) and electron-
temperature (eV) grids.  This module validates a checksummed manifest, parses
the native files, performs bilinear interpolation in log space, and converts
rate/power coefficients to SI.  It does not silently extrapolate beyond the
published grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..config import AtomicSpeciesConfig


_FLOAT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")
_RATE_KINDS = {"scd", "acd", "ccd", "qcd", "xcd"}
_POWER_KINDS = {"plt", "prb", "prc"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbers(line: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT.findall(line)]


@dataclass(frozen=True)
class ADF11Table:
    process: str
    element: str
    nuclear_charge: int
    log10_ne_cm3: np.ndarray
    log10_Te_eV: np.ndarray
    log10_coefficient_cgs: np.ndarray
    source_path: Path
    sha256: str

    def __post_init__(self) -> None:
        density = np.asarray(self.log10_ne_cm3, dtype=float)
        temperature = np.asarray(self.log10_Te_eV, dtype=float)
        values = np.asarray(self.log10_coefficient_cgs, dtype=float)
        expected = (self.nuclear_charge, len(temperature), len(density))
        if density.ndim != 1 or temperature.ndim != 1 or values.shape != expected:
            raise ValueError(f"ADF11 values must have shape {expected}")
        if np.any(np.diff(density) <= 0.0) or np.any(np.diff(temperature) <= 0.0):
            raise ValueError("ADF11 density and temperature grids must increase")
        if np.any(~np.isfinite(values)):
            raise ValueError("ADF11 coefficient grid contains non-finite values")
        object.__setattr__(self, "log10_ne_cm3", density)
        object.__setattr__(self, "log10_Te_eV", temperature)
        object.__setattr__(self, "log10_coefficient_cgs", values)

    @property
    def si_unit(self) -> str:
        if self.process in _RATE_KINDS:
            return "m^3 s^-1"
        if self.process in _POWER_KINDS:
            return "W m^3"
        raise ValueError(f"unsupported ADF11 process {self.process!r}")

    def coefficient_si(
        self,
        Te_eV: float | np.ndarray,
        ne_m3: float | np.ndarray,
        *,
        charge_index: int = 0,
        bounds: str = "raise",
    ) -> np.ndarray | float:
        if not 0 <= charge_index < self.nuclear_charge:
            raise ValueError("charge_index is outside the ADF11 charge blocks")
        temperature = np.asarray(Te_eV, dtype=float)
        density = np.asarray(ne_m3, dtype=float)
        temperature, density = np.broadcast_arrays(temperature, density)
        if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(density)):
            raise ValueError("Te and ne must be finite")
        if np.any(temperature <= 0.0) or np.any(density <= 0.0):
            raise ValueError("Te and ne must be positive")
        log_t = np.log10(temperature)
        log_n = np.log10(density / 1.0e6)
        if bounds not in {"raise", "clip"}:
            raise ValueError("bounds must be 'raise' or 'clip'")
        outside = (
            (log_t < self.log10_Te_eV[0])
            | (log_t > self.log10_Te_eV[-1])
            | (log_n < self.log10_ne_cm3[0])
            | (log_n > self.log10_ne_cm3[-1])
        )
        if bounds == "raise" and np.any(outside):
            raise ValueError("requested Te/ne lies outside the ADF11 grid")
        log_t = np.clip(log_t, self.log10_Te_eV[0], self.log10_Te_eV[-1])
        log_n = np.clip(log_n, self.log10_ne_cm3[0], self.log10_ne_cm3[-1])
        interpolator = RegularGridInterpolator(
            (self.log10_Te_eV, self.log10_ne_cm3),
            self.log10_coefficient_cgs[charge_index],
            bounds_error=True,
        )
        points = np.column_stack((log_t.ravel(), log_n.ravel()))
        coefficient = 10.0 ** interpolator(points).reshape(log_t.shape)
        # Both cm^3/s and W cm^3 acquire a 1e-6 volume conversion.
        coefficient_si = coefficient * 1.0e-6
        return float(coefficient_si) if coefficient_si.ndim == 0 else coefficient_si


def read_adf11(path: str | Path, *, process: str | None = None) -> ADF11Table:
    source = Path(path)
    lines = source.read_text(encoding="ascii", errors="strict").splitlines()
    if len(lines) < 5:
        raise ValueError("ADF11 file is too short")
    header = lines[0].split("/")
    integers = [int(value) for value in header[0].split()[:3]]
    if len(integers) != 3:
        raise ValueError("ADF11 header must contain nuclear charge, density count and temperature count")
    nuclear_charge, density_count, temperature_count = integers
    element = header[1].strip() if len(header) > 1 else "unknown"
    kind = (process or source.name[:3]).lower()
    if kind not in _RATE_KINDS | _POWER_KINDS:
        raise ValueError(f"unsupported ADF11 process {kind!r}")

    cursor = 2
    density_values: list[float] = []
    while cursor < len(lines) and len(density_values) < density_count:
        density_values.extend(_numbers(lines[cursor]))
        cursor += 1
    temperature_values: list[float] = []
    while cursor < len(lines) and len(temperature_values) < temperature_count:
        temperature_values.extend(_numbers(lines[cursor]))
        cursor += 1
    if len(density_values) != density_count or len(temperature_values) != temperature_count:
        raise ValueError("ADF11 coordinate grids are incomplete")

    blocks: list[np.ndarray] = []
    for _ in range(nuclear_charge):
        while cursor < len(lines) and "IPRT" not in lines[cursor]:
            cursor += 1
        if cursor >= len(lines):
            raise ValueError("ADF11 charge-block header is missing")
        cursor += 1
        values: list[float] = []
        required = density_count * temperature_count
        while cursor < len(lines) and len(values) < required:
            values.extend(_numbers(lines[cursor]))
            cursor += 1
        if len(values) != required:
            raise ValueError("ADF11 charge block is incomplete or overfull")
        blocks.append(np.asarray(values, dtype=float).reshape(temperature_count, density_count))
    return ADF11Table(
        kind,
        element,
        nuclear_charge,
        np.asarray(density_values, dtype=float),
        np.asarray(temperature_values, dtype=float),
        np.stack(blocks),
        source.resolve(),
        _sha256(source),
    )


def load_openadas_manifest(path: str | Path) -> tuple[dict[str, ADF11Table], Mapping[str, object]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("OpenADAS manifest must contain files")
    tables: dict[str, ADF11Table] = {}
    for process, record in files.items():
        source = manifest_path.parent / str(record["file"])
        actual = _sha256(source)
        expected = str(record["sha256"]).lower()
        if actual != expected:
            raise ValueError(f"checksum mismatch for OpenADAS {process}")
        tables[str(process)] = read_adf11(source, process=str(process))
    return tables, payload


class OpenADASAtomicRateProvider:
    """Hydrogen ADF11 provider compatible with ``DYONSpeciesWallModel``."""

    def __init__(self, manifest_path: str | Path, *, bounds: str = "raise") -> None:
        self.tables, self.manifest = load_openadas_manifest(manifest_path)
        missing = {"scd", "acd", "plt", "prb"} - set(self.tables)
        if missing:
            raise ValueError(f"OpenADAS manifest lacks {sorted(missing)}")
        if bounds not in {"raise", "clip"}:
            raise ValueError("bounds must be 'raise' or 'clip'")
        self.bounds = bounds
        self.label = f"OpenADAS:{self.manifest.get('dataset', 'ADF11')}"

    @staticmethod
    def _hydrogen(species: AtomicSpeciesConfig) -> None:
        if species.symbol not in {"H", "D", "T"} or species.max_charge != 1:
            raise ValueError("the bundled ADF11 snapshot is valid only for H/D/T charge states 0/1")

    def ionization_rate_m3_s(self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19) -> float:
        self._hydrogen(species)
        if charge != 0:
            raise ValueError("hydrogen ionization requires charge=0")
        return float(self.tables["scd"].coefficient_si(Te_eV, ne_m3, charge_index=0, bounds=self.bounds))

    def recombination_rate_m3_s(self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19) -> float:
        self._hydrogen(species)
        if charge != 1:
            raise ValueError("hydrogen recombination requires charge=1")
        return float(self.tables["acd"].coefficient_si(Te_eV, ne_m3, charge_index=0, bounds=self.bounds))

    def radiation_coefficient_W_m3(self, species: AtomicSpeciesConfig, charge: int, Te_eV: float, ne_m3: float = 1.0e19) -> float:
        self._hydrogen(species)
        process = "plt" if charge == 0 else "prb"
        return float(self.tables[process].coefficient_si(Te_eV, ne_m3, charge_index=0, bounds=self.bounds))

    def ionization_energy_eV(self, species: AtomicSpeciesConfig, charge: int) -> float:
        self._hydrogen(species)
        if charge != 0:
            raise ValueError("hydrogen ionization energy requires charge=0")
        if species.ionization_energies_eV is not None:
            return float(species.ionization_energies_eV[0])
        return 13.598434599702


__all__ = [
    "ADF11Table",
    "read_adf11",
    "load_openadas_manifest",
    "OpenADASAtomicRateProvider",
]
