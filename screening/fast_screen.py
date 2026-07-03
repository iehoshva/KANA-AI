"""Fast two-stage screening for laptop-scale performance.

Stage 1: Coarse scan (1T × 1ratio per pair) → rank all pairs
Stage 2: Full grid only for top-N pairs → detailed results

Performance: ~3 min for 160 compounds on a mid-range laptop.
"""

import time
from typing import List, Optional, Tuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from kana.config import Config, PipelineConfig, PRESETS
from kana.inference import KANAInference
from database.metadata_db import MetadataDB
from database.properties_db import PropertiesDB
from screening.selectivity import SelectivityComputer
from screening.des_validator import DESValidator
from screening.ranking import RankingEngine, ScreeningResult


# Default coarse grid
COARSE_T = np.array([298.15], dtype='float32')  # Room temperature only
COARSE_RATIO = np.array([0.50], dtype='float32')  # Equimolar only


def fast_screen(
    inference: KANAInference,
    pipe_cfg: PipelineConfig,
    db_meta: MetadataDB,
    db_props: PropertiesDB,
    target_code: str,
    impurity_code: str,
    target_smiles: str,
    impurity_smiles: str,
    mode: str = 'DES',
    top_n: int = 50,
    chunk_size: int = 500,
    verbose: bool = False,
) -> List[ScreeningResult]:
    """Two-stage fast screening.

    Args:
        inference: loaded KANAInference model
        pipe_cfg: pipeline configuration
        db_meta: metadata database
        db_props: properties database
        target_code: compound code for target
        impurity_code: compound code for impurity
        target_smiles: SMILES for target
        impurity_smiles: SMILES for impurity
        mode: 'DES' or 'ABS'
        top_n: number of top pairs to refine in Stage 2
        chunk_size: batch size for JAX inference
        verbose: print progress

    Returns:
        List of ScreeningResult for all evaluated systems
    """
    cfg = inference.cfg
    sel_comp = SelectivityComputer()
    des_validator = DESValidator(pipe_cfg, db_props)

    # Load all features
    all_feats = db_meta.get_all_with_features()
    target_feats = db_meta.get_features(target_code)
    impurity_feats = db_meta.get_features(impurity_code)

    if target_feats is None or impurity_feats is None:
        print("ERROR: Missing features for target or impurity")
        return []

    # Scale all features at once (pre-compute)
    scaled = {}
    for code, (sigma, scalars) in all_feats.items():
        if code in (target_code, impurity_code):
            continue
        s_sig, s_sca = inference.scale_features(
            sigma.reshape(1, -1), scalars.reshape(1, -1)
        )
        scaled[code] = (s_sig[0], s_sca[0])  # remove batch dim

    t_sig_s, t_sca_s = inference.scale_features(
        target_feats[0].reshape(1, -1), target_feats[1].reshape(1, -1)
    )
    i_sig_s, i_sca_s = inference.scale_features(
        impurity_feats[0].reshape(1, -1), impurity_feats[1].reshape(1, -1)
    )
    t_sig, t_sca = t_sig_s[0], t_sca_s[0]
    i_sig, i_sca = i_sig_s[0], i_sca_s[0]

    # Water for ABS
    w_sig, w_sca = None, None
    if mode == 'ABS':
        w_feats = db_meta.get_features('WATER')
        if w_feats is not None:
            ws, wc = inference.scale_features(
                w_feats[0].reshape(1, -1), w_feats[1].reshape(1, -1)
            )
            w_sig, w_sca = ws[0], wc[0]

    solvent_codes = list(scaled.keys())
    n_solvents = len(solvent_codes)

    if verbose:
        print(f"  Solvent candidates: {n_solvents}")
        print(f"  Stage 1: Coarse scan of {n_solvents**2} pairs...")

    # ================================================================
    # STAGE 1: Coarse scan — 1T × 1ratio per pair
    # ================================================================
    stage1_start = time.time()
    coarse_results = _coarse_scan(
        inference, cfg, scaled, t_sig, t_sca, i_sig, i_sca,
        w_sig, w_sca, solvent_codes, mode, chunk_size, verbose,
    )
    stage1_time = time.time() - stage1_start

    if verbose:
        print(f"  Stage 1 complete in {stage1_time:.1f}s")

    # Rank by coarse selectivity, pick top-N pairs
    coarse_arr = np.array(coarse_results)  # (n_pairs, 3) = [hba_idx, hbd_idx, S]
    if len(coarse_arr) == 0:
        return []

    # Sort by selectivity descending
    sorted_idx = np.argsort(-coarse_arr[:, 2])
    top_pairs_idx = sorted_idx[:top_n]
    top_pairs = coarse_arr[top_pairs_idx].astype(int)

    if verbose:
        print(f"  Top-{top_n} pairs: S range [{coarse_arr[top_pairs_idx[-1], 2]:.2f}, "
              f"{coarse_arr[top_pairs_idx[0], 2]:.2f}]")
        print(f"  Stage 2: Full grid for {len(top_pairs)} pairs...")

    # ================================================================
    # STAGE 2: Full grid for top-N pairs
    # ================================================================
    stage2_start = time.time()
    T_sweep = np.arange(pipe_cfg.T_min, pipe_cfg.T_max + 1, pipe_cfg.T_step, dtype='float32')
    ratio_sweep = np.array([
        0.99, 0.98, 0.95, 0.91, 0.83, 0.67, 0.50,
        0.33, 0.17, 0.09, 0.05, 0.02, 0.01,
    ], dtype='float32')

    all_results = []

    for pair_idx, (hba_i, hbd_i, _) in enumerate(top_pairs):
        hba_code = solvent_codes[hba_i]
        hbd_code = solvent_codes[hbd_i]

        hba_sig, hba_sca = scaled[hba_code]
        hbd_sig, hbd_sca = scaled[hbd_code]

        # Build full grid for this pair
        pair_results = _full_grid_pair(
            inference, cfg, pipe_cfg,
            hba_code, hbd_code, hba_sig, hba_sca, hbd_sig, hbd_sca,
            t_sig, t_sca, i_sig, i_sca, w_sig, w_sca,
            db_meta, db_props, des_validator, sel_comp,
            target_smiles, impurity_smiles,
            T_sweep, ratio_sweep, mode, chunk_size,
        )
        all_results.extend(pair_results)

        if verbose and (pair_idx + 1) % 10 == 0:
            print(f"    Processed {pair_idx + 1}/{len(top_pairs)} pairs...")

    stage2_time = time.time() - stage2_start
    if verbose:
        print(f"  Stage 2 complete in {stage2_time:.1f}s")
        print(f"  Total: {len(all_results)} systems evaluated")

    return all_results


def _coarse_scan(
    inference: KANAInference,
    cfg: Config,
    scaled: dict,
    t_sig: np.ndarray, t_sca: np.ndarray,
    i_sig: np.ndarray, i_sca: np.ndarray,
    w_sig: Optional[np.ndarray], w_sca: Optional[np.ndarray],
    solvent_codes: list,
    mode: str,
    chunk_size: int,
    verbose: bool,
) -> list:
    """Stage 1: Coarse scan of all solvent pairs at 1T × 1ratio."""
    max_n = cfg.MAX_COMPONENTS
    is_abs = mode == 'ABS'
    n_comp = 5 if is_abs else 4

    # Pre-build component feature arrays (shared across all pairs)
    t_sigma_padded = np.zeros((max_n, cfg.SIGMA_DIM, 1), dtype='float32')
    t_scalar_padded = np.zeros((max_n, cfg.SCALAR_DIM), dtype='float32')
    t_sigma_padded[2] = t_sig.reshape(cfg.SIGMA_DIM, 1)
    t_scalar_padded[2] = t_sca

    i_sigma_padded = np.zeros((max_n, cfg.SIGMA_DIM, 1), dtype='float32')
    i_scalar_padded = np.zeros((max_n, cfg.SCALAR_DIM), dtype='float32')
    i_sigma_padded[3] = i_sig.reshape(cfg.SIGMA_DIM, 1)
    i_scalar_padded[3] = i_sca

    if is_abs and w_sig is not None:
        w_sigma_padded = np.zeros((max_n, cfg.SIGMA_DIM, 1), dtype='float32')
        w_scalar_padded = np.zeros((max_n, cfg.SCALAR_DIM), dtype='float32')
        w_sigma_padded[4] = w_sig.reshape(cfg.SIGMA_DIM, 1)
        w_scalar_padded[4] = w_sca

    T_val = 298.15
    x_hba = 0.5
    x_hbd = 0.5

    # Build ALL pairs at once
    n_solvents = len(solvent_codes)
    n_pairs = n_solvents * n_solvents

    results = []

    # Process in chunks to avoid OOM
    for chunk_start in range(0, n_pairs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_pairs)
        chunk_len = chunk_end - chunk_start

        sigma_batch = np.zeros((chunk_len, max_n, cfg.SIGMA_DIM, 1), dtype='float32')
        scalar_batch = np.zeros((chunk_len, max_n, cfg.SCALAR_DIM), dtype='float32')
        mask_batch = np.zeros((chunk_len, max_n), dtype=bool)
        T_batch = np.full(chunk_len, T_val, dtype='float32')
        x_batch = np.zeros((chunk_len, max_n), dtype='float32')
        n_batch = np.zeros((chunk_len, max_n), dtype='float32')

        pair_indices = []

        for local_i in range(chunk_len):
            global_i = chunk_start + local_i
            hba_i = global_i // n_solvents
            hbd_i = global_i % n_solvents

            hba_code = solvent_codes[hba_i]
            hbd_code = solvent_codes[hbd_i]
            hba_sig, hba_sca = scaled[hba_code]
            hbd_sig, hbd_sca = scaled[hbd_code]

            # Build sigma tensor
            sigma_batch[local_i, 0, :, 0] = hba_sig.reshape(-1)
            sigma_batch[local_i, 1, :, 0] = hbd_sig.reshape(-1)
            sigma_batch[local_i, 2] = t_sigma_padded[2]
            sigma_batch[local_i, 3] = i_sigma_padded[3]
            if is_abs and w_sig is not None:
                sigma_batch[local_i, 4] = w_sigma_padded[4]

            # Build scalar tensor
            scalar_batch[local_i, 0] = hba_sca
            scalar_batch[local_i, 1] = hbd_sca
            scalar_batch[local_i, 2] = t_sca
            scalar_batch[local_i, 3] = i_sca
            if is_abs and w_sca is not None:
                scalar_batch[local_i, 4] = w_sca

            # Mask and composition
            mask_batch[local_i, :n_comp] = True
            x_batch[local_i, 0] = x_hba
            x_batch[local_i, 1] = x_hbd
            x_batch[local_i, 2] = 0.0  # infinite dilution
            x_batch[local_i, 3] = 0.0
            if is_abs:
                x_batch[local_i, 4] = 0.999
            n_batch[local_i] = x_batch[local_i]

            pair_indices.append((hba_i, hbd_i))

        # Forward pass (single JIT call)
        sigma_jax = jnp.array(sigma_batch)
        scalar_jax = jnp.array(scalar_batch)
        mask_jax = jnp.array(mask_batch)
        T_jax = jnp.array(T_batch)
        n_jax = jnp.array(n_batch)

        ln_gamma = inference.predict_ln_gamma(sigma_jax, scalar_jax, mask_jax, T_jax, n_jax)
        ln_gamma_np = np.array(ln_gamma)

        # Extract selectivity for each pair
        for local_i in range(chunk_len):
            lg_target = float(ln_gamma_np[local_i, 2])
            lg_impurity = float(ln_gamma_np[local_i, 3])
            S = float(np.exp(lg_impurity - lg_target))
            hba_i, hbd_i = pair_indices[local_i]
            results.append([hba_i, hbd_i, S])

        if verbose and chunk_start % (chunk_size * 5) == 0:
            print(f"    Coarse scan: {chunk_end}/{n_pairs} pairs...")

    return results


def _full_grid_pair(
    inference: KANAInference,
    cfg: Config,
    pipe_cfg: PipelineConfig,
    hba_code: str, hbd_code: str,
    hba_sig: np.ndarray, hba_sca: np.ndarray,
    hbd_sig: np.ndarray, hbd_sca: np.ndarray,
    t_sig: np.ndarray, t_sca: np.ndarray,
    i_sig: np.ndarray, i_sca: np.ndarray,
    w_sig: Optional[np.ndarray], w_sca: Optional[np.ndarray],
    db_meta: MetadataDB,
    db_props: PropertiesDB,
    des_validator: DESValidator,
    sel_comp: SelectivityComputer,
    target_smiles: str,
    impurity_smiles: str,
    T_sweep: np.ndarray,
    ratio_sweep: np.ndarray,
    mode: str,
    chunk_size: int,
) -> List[ScreeningResult]:
    """Stage 2: Full T×ratio grid for a single solvent pair."""
    max_n = cfg.MAX_COMPONENTS
    is_abs = mode == 'ABS'
    n_comp = 5 if is_abs else 4
    n_systems = len(T_sweep) * len(ratio_sweep)

    # Pre-allocate
    sigma_batch = np.zeros((n_systems, max_n, cfg.SIGMA_DIM, 1), dtype='float32')
    scalar_batch = np.zeros((n_systems, max_n, cfg.SCALAR_DIM), dtype='float32')
    mask_batch = np.zeros((n_systems, max_n), dtype=bool)
    T_batch = np.zeros(n_systems, dtype='float32')
    x_batch = np.zeros((n_systems, max_n), dtype='float32')

    # Fill component features (same for all systems in this pair)
    for sys_i in range(n_systems):
        sigma_batch[sys_i, 0, :, 0] = hba_sig.reshape(-1)
        sigma_batch[sys_i, 1, :, 0] = hbd_sig.reshape(-1)
        sigma_batch[sys_i, 2, :, 0] = t_sig.reshape(-1)
        sigma_batch[sys_i, 3, :, 0] = i_sig.reshape(-1)
        if is_abs and w_sig is not None:
            sigma_batch[sys_i, 4, :, 0] = w_sig.reshape(-1)

        scalar_batch[sys_i, 0] = hba_sca
        scalar_batch[sys_i, 1] = hbd_sca
        scalar_batch[sys_i, 2] = t_sca
        scalar_batch[sys_i, 3] = i_sca
        if is_abs and w_sca is not None:
            scalar_batch[sys_i, 4] = w_sca

        mask_batch[sys_i, :n_comp] = True

    # Fill T and ratio grid
    sys_i = 0
    for T in T_sweep:
        for r in ratio_sweep:
            T_batch[sys_i] = T
            x_batch[sys_i, 0] = r
            x_batch[sys_i, 1] = 1.0 - r
            if is_abs:
                x_batch[sys_i, 4] = 0.999
            sys_i += 1

    n_batch = x_batch.copy()

    # Batched inference (process in chunks)
    all_ln_gamma = []
    for chunk_start in range(0, n_systems, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_systems)
        lg = inference.predict_ln_gamma(
            jnp.array(sigma_batch[chunk_start:chunk_end]),
            jnp.array(scalar_batch[chunk_start:chunk_end]),
            jnp.array(mask_batch[chunk_start:chunk_end]),
            jnp.array(T_batch[chunk_start:chunk_end]),
            jnp.array(n_batch[chunk_start:chunk_end]),
        )
        all_ln_gamma.append(np.array(lg))

    ln_gamma_np = np.concatenate(all_ln_gamma, axis=0)

    # Build results
    hba_name = db_props.get_compound_name(hba_code)
    hbd_name = db_props.get_compound_name(hbd_code)
    hba_smiles = db_meta.get_smiles(hba_code) or ''
    hbd_smiles = db_meta.get_smiles(hbd_code) or ''

    results = []
    sys_i = 0
    for T in T_sweep:
        for r in ratio_sweep:
            lg_target = float(ln_gamma_np[sys_i, 2])
            lg_impurity = float(ln_gamma_np[sys_i, 3])
            S = float(np.exp(lg_impurity - lg_target))
            C = 1.0 / float(np.exp(lg_target))

            unc = sel_comp.propagate_uncertainty(np.array([S]), pipe_cfg.MAE_ln_gamma)

            # DES validation (only for DES mode)
            des_valid = False
            delta_Tf = 0.0
            T_eutectic = 0.0
            if mode == 'DES':
                des_val = des_validator.validate(
                    hba_code, hbd_code, lg_target, lg_impurity, float(r), float(T),
                )
                des_valid = des_val.valid
                delta_Tf = des_val.delta_Tf
                T_eutectic = des_val.T_eutectic

            # LLE validation (for ABS mode) — mark as valid by default
            # Full LLE proving (Michelsen TPD, tie-lines) is in lle_solver.py
            lle_valid = True if mode == 'ABS' else False

            results.append(ScreeningResult(
                hba_code=hba_code,
                hbd_code=hbd_code,
                hba_name=hba_name,
                hbd_name=hbd_name,
                hba_smiles=hba_smiles,
                hbd_smiles=hbd_smiles,
                T_opt_K=float(T),
                HBA_ratio=float(r),
                HBD_ratio=float(1.0 - r),
                S_inf=S,
                S_inf_lower95=float(unc['S_lower95'][0]),
                S_inf_upper95=float(unc['S_upper95'][0]),
                ln_gamma_target_inf=lg_target,
                ln_gamma_impurity_inf=lg_impurity,
                capacity_inf=C,
                des_valid=des_valid,
                delta_Tf_eutectic_K=delta_Tf,
                T_eutectic_K=T_eutectic,
                lle_valid=lle_valid,
                confidence=unc['confidence'][0],
                mode=mode,
                n_components=n_comp,
            ))
            sys_i += 1

    return results
