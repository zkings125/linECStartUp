"""Quasineutral and ambipolar electron-wall-loss closures."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..config import ChargeClosureConfig


@dataclass(frozen=True)
class ChargeClosureTerms:
    electron_wall_loss_m3_s: float
    free_electron_wall_loss_m3_s: float
    target_electron_wall_loss_m3_s: float
    positive_charge_wall_loss_m3_s: float
    ambipolar_potential_V: float
    ambipolar_potential_over_Te: float
    charge_continuity_residual_m3_s: float
    flux_limited: bool
    mode: str

    def diagnostics(self) -> dict[str, float | str]:
        residual_scale = max(
            abs(self.electron_wall_loss_m3_s),
            abs(self.target_electron_wall_loss_m3_s),
            abs(self.positive_charge_wall_loss_m3_s),
            1.0,
        )
        return {
            "charge_closure_mode": self.mode,
            "S_e_wall_free_m3_s": self.free_electron_wall_loss_m3_s,
            "S_e_wall_target_m3_s": self.target_electron_wall_loss_m3_s,
            "S_positive_charge_wall_m3_s": self.positive_charge_wall_loss_m3_s,
            "ambipolar_potential_V": self.ambipolar_potential_V,
            "ambipolar_potential_over_Te": self.ambipolar_potential_over_Te,
            "charge_continuity_residual_m3_s": self.charge_continuity_residual_m3_s,
            "charge_continuity_relative": self.charge_continuity_residual_m3_s / residual_scale,
            "ambipolar_flux_limited": float(self.flux_limited),
        }


class ChargeClosureModel:
    """Close net electron boundary loss against positive-charge loss.

    The sign convention uses positive values for loss from the plasma.  In
    exact quasineutral mode the effective boundary exchange may be negative if
    the requested source terms cannot be balanced by a positive wall loss.
    Ambipolar mode remains a positive Boltzmann-suppressed flux and reports a
    residual whenever the configured potential bounds prevent exact matching.
    """

    def __init__(self, config: ChargeClosureConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        Te_eV: float,
        free_electron_wall_loss_m3_s: float,
        positive_charge_wall_loss_m3_s: float,
        secondary_electron_source_m3_s: float,
        attachment_loss_m3_s: float,
    ) -> ChargeClosureTerms:
        Te = max(float(Te_eV), 1.0e-3)
        free_loss = max(float(free_electron_wall_loss_m3_s), 0.0)
        positive_loss = max(float(positive_charge_wall_loss_m3_s), 0.0)
        secondary = max(float(secondary_electron_source_m3_s), 0.0)
        attachment = max(float(attachment_loss_m3_s), 0.0)
        # dne-d(sum q*n_q)=secondary-attachment-S_e_wall+S_q,wall.
        target = positive_loss + secondary - attachment
        mode = self.config.mode
        potential_over_Te = 0.0
        potential_V = 0.0
        limited = False

        if mode == "legacy":
            effective_loss = free_loss
        elif mode == "quasineutral":
            # Signed net exchange is intentional: it is the algebraic boundary
            # term required by charge continuity, not necessarily a gross flux.
            effective_loss = target
            if target > 0.0 and free_loss > 0.0:
                potential_over_Te = math.log(free_loss / target)
                potential_V = Te * potential_over_Te
        elif mode == "ambipolar":
            flux_floor = float(self.config.flux_floor_m3_s)
            if free_loss <= 0.0:
                potential_over_Te = self.config.potential_min_over_Te
                effective_loss = 0.0
                limited = target > flux_floor
            else:
                requested = max(target, flux_floor)
                unconstrained = math.log(free_loss / requested)
                potential_over_Te = min(
                    self.config.potential_max_over_Te,
                    max(self.config.potential_min_over_Te, unconstrained),
                )
                limited = (target <= 0.0) or not math.isclose(
                    potential_over_Te, unconstrained, rel_tol=1.0e-12, abs_tol=1.0e-14
                )
                effective_loss = free_loss * math.exp(-potential_over_Te)
            potential_V = Te * potential_over_Te
        else:  # Pydantic prevents this, but keep provider failures explicit.
            raise ValueError(f"未知 charge closure mode: {mode}")

        residual = secondary - attachment - effective_loss + positive_loss
        return ChargeClosureTerms(
            electron_wall_loss_m3_s=effective_loss,
            free_electron_wall_loss_m3_s=free_loss,
            target_electron_wall_loss_m3_s=target,
            positive_charge_wall_loss_m3_s=positive_loss,
            ambipolar_potential_V=potential_V,
            ambipolar_potential_over_Te=potential_over_Te,
            charge_continuity_residual_m3_s=residual,
            flux_limited=limited,
            mode=mode,
        )
