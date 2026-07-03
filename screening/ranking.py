"""Filtering cascade and Pareto-optimal ranking engine."""

from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kana.config import PipelineConfig


@dataclass
class ScreeningResult:
    """Single screening result for one (HBA, HBD, T, ratio) system."""
    hba_code: str
    hbd_code: str
    hba_name: str
    hbd_name: str
    hba_smiles: str
    hbd_smiles: str
    T_opt_K: float
    HBA_ratio: float
    HBD_ratio: float

    # Infinite dilution
    S_inf: float
    S_inf_lower95: float
    S_inf_upper95: float
    ln_gamma_target_inf: float
    ln_gamma_impurity_inf: float
    capacity_inf: float

    # Finite loading (if computed)
    S_finite_z001: float = 0.0
    S_finite_z001_lower95: float = 0.0
    S_finite_z001_upper95: float = 0.0
    K_target_z001: float = 0.0
    K_target_zinf: float = 0.0

    # LLE validity
    phase_stable: bool = False
    tpd_min: float = 0.0
    isoactivity_converged: bool = False
    gibbs_mixing_concave: bool = False
    lle_valid: bool = False

    # DES validation
    delta_Tf_eutectic_K: float = 0.0
    T_eutectic_K: float = 0.0
    des_valid: bool = False

    # Extraction design
    N_stages_99pct: float = 0.0
    S_F_optimal: float = 0.0
    K_loading_retention_ratio: float = 0.0

    # Meta
    confidence: str = 'LOW'
    pareto_optimal: bool = False
    mode: str = 'DES'
    n_components: int = 4


class RankingEngine:
    """7-gate filtering cascade and Pareto-optimal ranking."""

    def __init__(self, pipe_cfg: PipelineConfig):
        self.pipe_cfg = pipe_cfg

    def filter_and_rank(self, results: List[ScreeningResult],
                        mode: str) -> pd.DataFrame:
        """Apply filtering cascade and rank results.

        Filtering cascade:
        1. Remove non-LLE-valid systems (ABS mode)
        2. Remove DES failures (DES mode)
        3. Remove S_finite_upper95 < 1
        4. Remove S_finite_lower95 < S_threshold
        5. Remove C_target < 1e-5
        6. Compute Pareto frontier
        7. Rank by finite-loading selectivity

        Returns:
            Ranked DataFrame
        """
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([vars(r) for r in results])

        # Add clean display columns (names without underscores)
        if 'hba_name' in df.columns:
            df['solvent_A'] = df['hba_name'].str.replace('_', ' ')
            df['solvent_B'] = df['hbd_name'].str.replace('_', ' ')
            df['solvent_A_code'] = df['hba_code']
            df['solvent_B_code'] = df['hbd_code']
            df['ratio_A'] = df['HBA_ratio']
            df['ratio_B'] = df['HBD_ratio']

        # Deduplicate symmetric pairs (A+B = B+A due to permutation invariance)
        # Create a canonical pair key: sorted code pair
        df['pair_key'] = df.apply(
            lambda r: tuple(sorted([r['hba_code'], r['hbd_code']])), axis=1
        )
        # Keep the first occurrence of each (pair, T, ratio) combination
        df = df.drop_duplicates(subset=['pair_key', 'T_opt_K', 'HBA_ratio'], keep='first')
        df = df.drop(columns=['pair_key'])

        S_threshold = self.pipe_cfg.S_min_abs if mode == 'ABS' else self.pipe_cfg.S_min_des

        # Gate 1: LLE validity (ABS mode)
        if mode == 'ABS':
            df = df[df['lle_valid']]

        # Gate 2: DES validity (DES mode)
        if mode == 'DES':
            df = df[df['des_valid']]

        # Gate 3: Upper confidence bound > 1
        df = df[df['S_inf_upper95'] >= 1]

        # Gate 4: Lower confidence bound > threshold
        df = df[df['S_inf_lower95'] >= S_threshold]

        # Gate 5: Minimum capacity
        df = df[df['capacity_inf'] >= 1e-5]

        if len(df) == 0:
            return df

        # Gate 6: Pareto frontier on (Selectivity, Capacity)
        df = self._compute_pareto(df)

        # Gate 7: Rank by selectivity (descending)
        df = df.sort_values('S_inf', ascending=False)
        df['rank'] = range(1, len(df) + 1)

        return df

    def _compute_pareto(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pareto-optimal frontier on (S_inf, capacity_inf)."""
        if len(df) == 0:
            return df

        # Sort by selectivity descending
        sorted_df = df.sort_values('S_inf', ascending=False).copy()

        pareto_mask = np.zeros(len(sorted_df), dtype=bool)
        max_capacity = -np.inf

        for i, (_, row) in enumerate(sorted_df.iterrows()):
            if row['capacity_inf'] >= max_capacity:
                pareto_mask[i] = True
                max_capacity = row['capacity_inf']

        sorted_df['pareto_optimal'] = pareto_mask
        return sorted_df

    def assign_confidence(self, S: float, S_lower: float, S_upper: float) -> str:
        """Assign confidence tier based on uncertainty bounds."""
        if S_lower >= 1:
            return 'HIGH'
        elif S_upper >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'

    def selectivity_retention(self, S_inf: float, S_finite: float) -> float:
        """Compute selectivity retention ratio S_finite / S_inf."""
        if S_inf <= 0:
            return 0.0
        return S_finite / S_inf
