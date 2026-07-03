"""Multi-stage countercurrent extraction design (McCabe-Thiele / Kremser)."""

from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExtractionDesign:
    """Result of countercurrent extraction design."""
    K_avg: float
    S_F_optimal: float  # Solvent-to-feed ratio
    N_theoretical: float  # Number of theoretical stages
    E_factor: float  # Extraction factor
    feasible: bool


class CountercurrentDesigner:
    """McCabe-Thiele extraction stage calculation via Kremser equation."""

    def compute_stages(self, K_values: pd.DataFrame,
                       purity_target: float = 0.99) -> pd.DataFrame:
        """Compute theoretical stages for a range of solvent ratios.

        Kremser equation:
        N = ln[(x_in - x_in*/K) / (x_out - x_out*/K)] / ln(E)

        where E = K * (S/F) is the extraction factor.

        Args:
            K_values: DataFrame with columns 'z_target', 'K_target'
            purity_target: desired purity (default 0.99)

        Returns:
            DataFrame with S_F, N_theoretical, E, feasible columns
        """
        # Average K at low loading
        low_loading = K_values[K_values['z_target'] < 0.05]
        if len(low_loading) == 0:
            low_loading = K_values
        K_avg = float(low_loading['K_target'].mean())

        if K_avg <= 0 or np.isnan(K_avg):
            return pd.DataFrame(columns=['S_F', 'N_theoretical', 'E', 'feasible'])

        S_F_range = np.linspace(0.5, 10.0, 50)
        results = []

        for S_F in S_F_range:
            E = K_avg * S_F

            if abs(E - 1.0) < 1e-6:
                # E=1 is a special case: infinite stages needed
                N = np.inf
            elif E <= 0:
                N = np.inf
            else:
                # Kremser equation for countercurrent extraction
                # Assuming x_out = 0.01 (1% residual) and x_in* = 0
                x_in = 1.0 - purity_target  # solute in raffinate
                x_out = purity_target  # solute in extract

                if E < 1:
                    # E < 1: extraction is unfavorable
                    N = np.log((x_out - x_out / K_avg) /
                               (x_in - x_in / K_avg)) / np.log(E)
                else:
                    N = np.log((x_out / K_avg) / (x_in / K_avg)) / np.log(E)

            N = max(0, N)
            feasible = N <= 20 and E > 1.0

            results.append({
                'S_F': S_F,
                'N_theoretical': N,
                'E': E,
                'feasible': feasible,
            })

        return pd.DataFrame(results)

    def find_optimal_design(self, K_values: pd.DataFrame,
                            purity_target: float = 0.99) -> ExtractionDesign:
        """Find the optimal solvent-to-feed ratio.

        Minimizes stages while keeping E > 1 (feasible extraction).
        """
        stages_df = self.compute_stages(K_values, purity_target)

        feasible = stages_df[stages_df['feasible']]
        if len(feasible) == 0:
            low_loading = K_values[K_values['z_target'] < 0.05]
            K_avg = float(low_loading['K_target'].mean()) if len(low_loading) > 0 else 0.0
            return ExtractionDesign(
                K_avg=K_avg, S_F_optimal=0, N_theoretical=0,
                E_factor=0, feasible=False,
            )

        # Find minimum stages among feasible designs
        best = feasible.loc[feasible['N_theoretical'].idxmin()]
        K_avg = float(K_values[K_values['z_target'] < 0.05]['K_target'].mean())

        return ExtractionDesign(
            K_avg=K_avg,
            S_F_optimal=float(best['S_F']),
            N_theoretical=float(best['N_theoretical']),
            E_factor=float(best['E']),
            feasible=True,
        )
