"""Selectivity computation for DES and ABS extraction modes."""

import numpy as np
from typing import Optional


class SelectivityComputer:
    """Computes extraction selectivity and capacity from ln_gamma predictions."""

    @staticmethod
    def compute_des(ln_gamma: np.ndarray, target_idx: int,
                    impurity_idx: int) -> dict:
        """DES mode: Solid-Liquid Extraction.

        S = exp(ln_gamma_impurity - ln_gamma_target)
        C = 1 / exp(ln_gamma_target)

        Args:
            ln_gamma: shape (batch, max_n) predicted ln gamma values
            target_idx: index of target compound
            impurity_idx: index of impurity compound

        Returns:
            dict with selectivity, capacity, ln_gamma values
        """
        lg_target = ln_gamma[:, target_idx]
        lg_impurity = ln_gamma[:, impurity_idx]

        S = np.exp(lg_impurity - lg_target)
        C = 1.0 / np.exp(lg_target)

        return {
            'selectivity': S,
            'capacity': C,
            'ln_gamma_target': lg_target,
            'ln_gamma_impurity': lg_impurity,
        }

    @staticmethod
    def compute_abs(ln_gamma_aq: np.ndarray, ln_gamma_des: np.ndarray,
                    target_idx: int, impurity_idx: int) -> dict:
        """ABS mode: Aqueous Biphasic System.

        K_target = exp(ln_gamma_target_Aq - ln_gamma_target_DES)
        K_impurity = exp(ln_gamma_impurity_Aq - ln_gamma_impurity_DES)
        S = K_target / K_impurity

        Args:
            ln_gamma_aq: ln gamma in aqueous phase
            ln_gamma_des: ln gamma in DES-rich phase
            target_idx: index of target compound
            impurity_idx: index of impurity compound

        Returns:
            dict with selectivity, K values, ln_gamma values
        """
        lg_target_aq = ln_gamma_aq[:, target_idx]
        lg_impurity_aq = ln_gamma_aq[:, impurity_idx]
        lg_target_des = ln_gamma_des[:, target_idx]
        lg_impurity_des = ln_gamma_des[:, impurity_idx]

        K_target = np.exp(lg_target_aq - lg_target_des)
        K_impurity = np.exp(lg_impurity_aq - lg_impurity_des)

        S = K_target / np.maximum(K_impurity, 1e-30)

        return {
            'selectivity': S,
            'K_target': K_target,
            'K_impurity': K_impurity,
            'ln_gamma_target_aq': lg_target_aq,
            'ln_gamma_target_des': lg_target_des,
            'ln_gamma_impurity_aq': lg_impurity_aq,
            'ln_gamma_impurity_des': lg_impurity_des,
        }

    @staticmethod
    def propagate_uncertainty(S: np.ndarray, mae_ln_gamma: float) -> dict:
        """Propagate model uncertainty to selectivity.

        sigma_ln_S = sqrt(2) * MAE
        S_lower = exp(ln_S - 1.96 * sigma_ln_S)
        S_upper = exp(ln_S + 1.96 * sigma_ln_S)
        """
        ln_S = np.log(np.maximum(S, 1e-30))
        sigma_ln_S = mae_ln_gamma * np.sqrt(2)

        S_lower = np.exp(ln_S - 1.96 * sigma_ln_S)
        S_upper = np.exp(ln_S + 1.96 * sigma_ln_S)

        confidence = np.where(
            S_lower >= 1, 'HIGH',
            np.where(S_upper >= 1, 'MEDIUM', 'LOW'),
        )

        return {
            'S_lower95': S_lower,
            'S_upper95': S_upper,
            'sigma_ln_S': sigma_ln_S,
            'confidence': confidence,
        }
