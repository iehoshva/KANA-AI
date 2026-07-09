"""Combinatorial grid construction for batched inference."""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from kana.config import Config, PipelineConfig


# Hardcoded water features (always present for ABS mode)
# 10 features matching training data order:
#   [HOMO, LUMO, Dipole, Max_Charge, Min_Charge, Energy_Gap, 0, 0, 0, 0]
WATER_SCALARS_RAW = np.array([
    -12.62, 0.48,     # HOMO, LUMO
    1.85,              # Dipole
    0.42, -0.42,       # Max_Charge, Min_Charge
    13.10,             # Energy_Gap (HOMO-LUMO)
    0.0, 0.0, 0.0, 0.0,
], dtype='float32')


@dataclass
class SystemSpec:
    """Specification for a single system point."""
    hba_code: str
    hbd_code: str
    target_idx: int
    impurity_idx: int
    hba_idx: int
    hbd_idx: int
    water_idx: int  # -1 if DES mode
    T: float
    x_hba: float
    x_hbd: float
    mode: str  # 'ABS' or 'DES'


class BatchBuilder:
    """Builds batched tensors for combinatorial screening."""

    def __init__(self, pipe_cfg: PipelineConfig, model_cfg: Config):
        self.pipe_cfg = pipe_cfg
        self.cfg = model_cfg
        self.max_n = model_cfg.MAX_COMPONENTS

    def build_t_sweep(self) -> np.ndarray:
        """Temperature sweep array."""
        return np.arange(
            self.pipe_cfg.T_min,
            self.pipe_cfg.T_max + 1,
            self.pipe_cfg.T_step,
            dtype='float32',
        )

    def build_ratio_sweep(self) -> np.ndarray:
        """HBA:HBD molar ratio sweep (logarithmic spacing)."""
        return np.array([
            0.99, 0.98, 0.95, 0.91, 0.83, 0.67, 0.50,
            0.33, 0.17, 0.09, 0.05, 0.02, 0.01,
        ], dtype='float32')

    def build_system_batch(
        self,
        hba_sigma: np.ndarray, hba_scalars: np.ndarray,
        hbd_sigma: np.ndarray, hbd_scalars: np.ndarray,
        target_sigma: np.ndarray, target_scalars: np.ndarray,
        impurity_sigma: np.ndarray, impurity_scalars: np.ndarray,
        water_sigma: Optional[np.ndarray],
        water_scalars: Optional[np.ndarray],
        mode: str,
        T_list: Optional[np.ndarray] = None,
        ratio_list: Optional[np.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
               jnp.ndarray, jnp.ndarray, jnp.ndarray, List[SystemSpec]]:
        """Build batched tensors for all (T, ratio) combinations.

        Returns:
            (sigma_batch, scalar_batch, mask_batch, T_batch, x_batch, n_batch, specs)
        """
        if T_list is None:
            T_list = self.build_t_sweep()
        if ratio_list is None:
            ratio_list = self.build_ratio_sweep()

        is_abs = mode == 'ABS'
        n_comp = 5 if is_abs else 4

        # Component order: [HBA, HBD, Target, Impurity, (Water)]
        sigmas_raw = [hba_sigma, hbd_sigma, target_sigma, impurity_sigma]
        scalars_raw = [hba_scalars, hbd_scalars, target_scalars, impurity_scalars]

        if is_abs and water_sigma is not None:
            sigmas_raw.append(water_sigma)
            scalars_raw.append(water_scalars if water_scalars is not None else WATER_SCALARS_RAW)

        all_sigmas = []
        all_scalars = []
        all_masks = []
        all_T = []
        all_x = []
        all_n = []
        specs = []

        for T in T_list:
            for r in ratio_list:
                x_hba = r
                x_hbd = 1.0 - r
                x_target = 0.0  # infinite dilution
                x_imp = 0.0

                x_vec = np.zeros(self.max_n, dtype='float32')
                n_vec = np.zeros(self.max_n, dtype='float32')

                if is_abs:
                    x_water = 0.999
                    x_vec_raw = np.array([x_hba, x_hbd, x_target, x_imp, x_water], dtype='float32')
                else:
                    x_vec_raw = np.array([x_hba, x_hbd, x_target, x_imp], dtype='float32')

                x_vec[:n_comp] = x_vec_raw
                n_vec[:n_comp] = x_vec_raw

                # Build sigma and scalar tensors
                sigma_tensor = np.zeros((self.max_n, self.cfg.SIGMA_DIM, 1), dtype='float32')
                scalar_tensor = np.zeros((self.max_n, self.cfg.SCALAR_DIM), dtype='float32')

                for i in range(n_comp):
                    sigma_tensor[i, :, 0] = sigmas_raw[i][:self.cfg.SIGMA_DIM]
                    scalar_tensor[i, :self.cfg.SCALAR_DIM] = scalars_raw[i][:self.cfg.SCALAR_DIM]

                mask = np.zeros(self.max_n, dtype=bool)
                mask[:n_comp] = True

                all_sigmas.append(sigma_tensor)
                all_scalars.append(scalar_tensor)
                all_masks.append(mask)
                all_T.append(T)
                all_x.append(x_vec)
                all_n.append(n_vec)

                specs.append(SystemSpec(
                    hba_code='', hbd_code='',
                    target_idx=2, impurity_idx=3,
                    hba_idx=0, hbd_idx=1,
                    water_idx=4 if is_abs else -1,
                    T=T, x_hba=x_hba, x_hbd=x_hbd, mode=mode,
                ))

        # Stack into JAX tensors
        sigma_batch = jnp.array(np.stack(all_sigmas))
        scalar_batch = jnp.array(np.stack(all_scalars))
        mask_batch = jnp.array(np.stack(all_masks))
        T_batch = jnp.array(np.array(all_T))
        x_batch = jnp.array(np.stack(all_x))
        n_batch = jnp.array(np.stack(all_n))

        return sigma_batch, scalar_batch, mask_batch, T_batch, x_batch, n_batch, specs
