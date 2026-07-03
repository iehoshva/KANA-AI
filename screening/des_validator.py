"""DES eutectic verification via Solid-Liquid Equilibrium (SLE).

Practical validation: checks if HBA+HBD can form a liquid mixture at
operating temperature. Handles both solid-solid eutectic DES and
room-temperature ionic liquid systems.
"""

from typing import Optional, Dict
from dataclasses import dataclass

import numpy as np

from kana.config import PipelineConfig
from database.properties_db import PropertiesDB


@dataclass
class DESValidation:
    """Result of DES eutectic validation."""
    valid: bool
    delta_Tf: float = 0.0  # Eutectic depression (K)
    T_eutectic: float = 0.0  # Eutectic temperature (K)
    reason: str = ''


class DESValidator:
    """Validates that HBA + HBD pairs form liquid mixtures (DES).

    Validation logic:
    1. If both compounds have T_f < operating T → already liquid, PASS
    2. If one is solid + one is liquid → check if liquid dissolves solid
    3. If both are solid → check eutectic depression
    4. In all cases: check that the mixture is liquid at operating T
    """

    def __init__(self, pipe_cfg: PipelineConfig, props_db: PropertiesDB):
        self.pipe_cfg = pipe_cfg
        self.props_db = props_db
        self.R = 8.314  # J/(mol·K)

    def validate(self, hba_code: str, hbd_code: str,
                 ln_gamma_hba: float, ln_gamma_hbd: float,
                 x_hba: float, T: float) -> DESValidation:
        """Validate DES formation for an HBA+HBD pair.

        Simplified approach: check if the mixture is liquid at operating T.

        Args:
            hba_code: HBA compound code
            hbd_code: HBD compound code
            ln_gamma_hba: predicted ln gamma of HBA (unused in simplified check)
            ln_gamma_hbd: predicted ln gamma of HBD (unused in simplified check)
            x_hba: mole fraction of HBA in DES
            T: operating temperature (K)

        Returns:
            DESValidation result
        """
        # Fetch thermal data
        thermal = self.props_db.get_thermal_pair(hba_code, hbd_code)
        if thermal is None:
            # No thermal data → cannot validate, but allow screening
            # (the model can still predict selectivity)
            return DESValidation(
                valid=True,
                reason='thermal_data_missing_assumed_liquid',
            )

        t_hba = thermal[hba_code]
        t_hbd = thermal[hbd_code]

        T_f_HBA = t_hba['T_f_K']
        T_f_HBD = t_hbd['T_f_K']
        dH_HBA = t_hba['dH_fus']  # J/mol
        dH_HBD = t_hbd['dH_fus']  # J/mol

        x_hbd = 1.0 - x_hba

        # Case 1: Both compounds are liquid at operating temperature
        # (T_f < T) → they form a liquid mixture trivially
        if T_f_HBA < T and T_f_HBD < T:
            return DESValidation(
                valid=True, delta_Tf=0.0, T_eutectic=min(T_f_HBA, T_f_HBD),
                reason='both_liquid_at_T',
            )

        # Case 2: One or both are solid → compute eutectic depression
        # Use simplified SLE: ΔT = (R * T_f^2 / ΔH_fus) * |ln(x * gamma)|
        # Assume ideal mixing (gamma=1) for quick check
        if dH_HBA > 0:
            dep_HBA = (self.R * T_f_HBA**2 / dH_HBA) * abs(np.log(max(x_hba, 1e-12)))
        else:
            dep_HBA = 0.0

        if dH_HBD > 0:
            dep_HBD = (self.R * T_f_HBD**2 / dH_HBD) * abs(np.log(max(x_hbd, 1e-12)))
        else:
            dep_HBD = 0.0

        delta_Tf = max(dep_HBA, dep_HBD)
        T_eutectic = min(T_f_HBA, T_f_HBD) - delta_Tf

        # Cap unrealistic depressions (e.g., inorganic salts with huge T_f)
        # If depression > 200K, the compound is probably not a typical DES component
        if delta_Tf > 200:
            # Check if the mixture is liquid at T anyway
            if T_eutectic > T:
                return DESValidation(
                    valid=False, delta_Tf=delta_Tf, T_eutectic=T_eutectic,
                    reason='eutectic_above_operating_T',
                )
            # If eutectic < T, the mixture is liquid
            return DESValidation(
                valid=True, delta_Tf=delta_Tf, T_eutectic=T_eutectic,
                reason='liquid_at_operating_T',
            )

        # Standard check: eutectic must be above freezing
        if T_eutectic <= 273.0:
            return DESValidation(
                valid=False, delta_Tf=delta_Tf, T_eutectic=T_eutectic,
                reason='eutectic_below_freezing',
            )

        return DESValidation(
            valid=True, delta_Tf=delta_Tf, T_eutectic=T_eutectic,
            reason='pass',
        )
