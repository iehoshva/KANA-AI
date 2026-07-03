"""End-to-end KANA extraction screening orchestrator.

Uses two-stage fast screening for laptop-scale performance:
  Stage 1: Coarse scan (1T × 1ratio) → rank all pairs (~2 min)
  Stage 2: Full grid for top-N pairs → detailed results (~1 min)
  Total: ~3 minutes for 160 compounds.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure package imports work when run as script
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))

from kana.config import PipelineConfig, Config, PRESETS, load_config
from kana.inference import KANAInference
from database.metadata_db import MetadataDB
from database.properties_db import PropertiesDB
from screening.fast_screen import fast_screen
from screening.selectivity import SelectivityComputer
from screening.countercurrent import CountercurrentDesigner
from screening.ranking import RankingEngine, ScreeningResult
from output.csv_export import CSVExporter
from output.report_gen import ReportGenerator
from output.visualization import Visualizer


def run_screening(
    target_smiles: str,
    impurity_smiles: str,
    mode: str = 'DES',
    config_path: Optional[str] = None,
    s_min: Optional[float] = None,
    output_dir: Optional[str] = None,
    target_code: Optional[str] = None,
    impurity_code: Optional[str] = None,
    top_n: int = 10,
    top_pairs: int = 50,
    no_plots: bool = False,
    verbose: bool = False,
    fill_thermal: bool = True,
):
    """Run complete KANA extraction screening pipeline.

    Args:
        target_smiles: SMILES of target compound
        impurity_smiles: SMILES of impurity compound
        mode: 'ABS' or 'DES'
        config_path: path to config.yaml
        s_min: minimum selectivity threshold
        output_dir: output directory override
        target_code: compound code for target in DB
        impurity_code: compound code for impurity in DB
        top_n: number of top systems in final report
        top_pairs: number of top pairs to refine in Stage 2
        no_plots: skip plot generation
        verbose: verbose output
        fill_thermal: auto-fill missing thermal data via estimation
    """

    start_time = time.time()

    # ================================================================
    # STAGE 0: ENVIRONMENT INITIALIZATION
    # ================================================================
    print("=" * 70)
    print("KANA — Kimia-informed Artificial Neural-network coffenovA")
    print("Post-Training Extraction Screening Pipeline v2.0 (Fast Mode)")
    print("=" * 70)

    pipe_cfg = load_config(config_path)
    if s_min is not None:
        if mode == 'ABS':
            object.__setattr__(pipe_cfg, 'S_min_abs', s_min)
        else:
            object.__setattr__(pipe_cfg, 'S_min_des', s_min)
    if output_dir:
        object.__setattr__(pipe_cfg, 'output_dir', Path(output_dir))
    pipe_cfg.ensure_dirs()

    print(f"\nMode: {mode}")
    print(f"Target: {target_smiles}")
    print(f"Impurity: {impurity_smiles}")
    print(f"Output: {pipe_cfg.output_dir}")

    # Load model
    print("\n[STAGE 0] Loading KANA model...")
    inference = KANAInference(pipe_cfg)
    inference.load()
    print("  Model loaded. Scalers applied. JIT warmup complete.")

    # Open databases
    db_meta_path = pipe_cfg.resolve(pipe_cfg.db_metadata).resolve()
    db_props_path = pipe_cfg.resolve(pipe_cfg.db_properties).resolve()
    db_meta = MetadataDB(db_meta_path)
    db_props = PropertiesDB(db_props_path)

    # ================================================================
    # STAGE 0B: Fill missing thermal data if needed
    # ================================================================
    if fill_thermal and mode == 'DES':
        from screening.thermal_estimator import fill_missing_thermal_data
        n_filled = fill_missing_thermal_data(db_props_path)
        if n_filled > 0:
            # Reopen DB to pick up changes
            db_props.close()
            db_props = PropertiesDB(db_props_path)

    # ================================================================
    # STAGE 1: INPUT DEFINITION — Resolve compound codes
    # ================================================================
    print("\n[STAGE 1] Resolving input compounds...")

    if target_code is None:
        target_code = _find_compound(target_smiles, db_meta, "TARGET")
    if impurity_code is None:
        impurity_code = _find_compound(impurity_smiles, db_meta, "IMPURITY")

    target_name = db_props.get_compound_name(target_code)
    impurity_name = db_props.get_compound_name(impurity_code)
    print(f"  Target: {target_code} ({target_name})")
    print(f"  Impurity: {impurity_code} ({impurity_name})")

    # Verify features exist
    if not db_meta.has_compound(target_code):
        print(f"  ERROR: No features for target {target_code}")
        return None
    if not db_meta.has_compound(impurity_code):
        print(f"  ERROR: No features for impurity {impurity_code}")
        return None

    # ================================================================
    # STAGE 2-4: FAST TWO-STAGE SCREENING
    # ================================================================
    print("\n[STAGE 2-4] Running fast two-stage screening...")
    print(f"  Stage 1: Coarse scan of all solvent pairs")
    print(f"  Stage 2: Full grid for top-{top_pairs} pairs")

    all_results = fast_screen(
        inference=inference,
        pipe_cfg=pipe_cfg,
        db_meta=db_meta,
        db_props=db_props,
        target_code=target_code,
        impurity_code=impurity_code,
        target_smiles=target_smiles,
        impurity_smiles=impurity_smiles,
        mode=mode,
        top_n=top_pairs,
        verbose=verbose,
    )

    if not all_results:
        print("\n  WARNING: No systems evaluated.")
        return None

    # ================================================================
    # STAGE 5-6: RANKING & FILTERING
    # ================================================================
    print("\n[STAGE 5-6] Filtering and ranking...")

    ranking_engine = RankingEngine(pipe_cfg)
    ranked_df = ranking_engine.filter_and_rank(all_results, mode)
    print(f"  Systems after filtering: {len(ranked_df)}")

    if len(ranked_df) == 0:
        print("\n  WARNING: No systems passed filtering criteria.")
        print("  Consider relaxing S_min threshold.")

    # ================================================================
    # STAGE 7: OUTPUT GENERATION
    # ================================================================
    print("\n[STAGE 7] Generating output...")

    csv_exporter = CSVExporter(pipe_cfg.output_dir)
    csv_path = csv_exporter.export(ranked_df)
    print(f"  CSV: {csv_path}")

    report_gen = ReportGenerator(pipe_cfg.output_dir)
    report_path = report_gen.generate(ranked_df, target_smiles, impurity_smiles, mode)
    print(f"  Report: {report_path}")

    # Plots
    if not no_plots and len(ranked_df) > 0:
        viz = Visualizer(pipe_cfg.output_dir)
        top = ranked_df.iloc[0]
        pair_df = ranked_df[
            (ranked_df['hba_code'] == top['hba_code']) &
            (ranked_df['hbd_code'] == top['hbd_code'])
        ]
        if len(pair_df) > 1:
            plot_path = viz.plot_selectivity_heatmap(
                pair_df, top['hba_code'], top['hbd_code'], mode
            )
            if plot_path:
                print(f"  Heatmap: {plot_path}")

    # ================================================================
    # DONE
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Screening complete in {elapsed:.1f}s")
    print(f"Results: {pipe_cfg.output_dir}")
    print(f"{'=' * 70}")

    if len(ranked_df) > 0:
        print(f"\nTop system: {ranked_df.iloc[0]['hba_name']} + {ranked_df.iloc[0]['hbd_name']}")
        print(f"  S_inf = {ranked_df.iloc[0]['S_inf']:.2f}")
        print(f"  Confidence: {ranked_df.iloc[0]['confidence']}")

    db_meta.close()
    db_props.close()

    return ranked_df


def _find_compound(smiles: str, db_meta: MetadataDB, label: str) -> str:
    """Find compound in DB by SMILES."""
    all_codes = db_meta.get_all_codes()
    for code in all_codes:
        db_smiles = db_meta.get_smiles(code)
        if db_smiles == smiles:
            return code

    print(f"  WARNING: {label} not found in DB by SMILES.")
    print(f"  Please provide --target-code and --impurity-code manually.")
    print(f"  Available codes: {all_codes[:10]}...")
    sys.exit(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--impurity', required=True)
    parser.add_argument('--mode', default='DES', choices=['ABS', 'DES'])
    parser.add_argument('--config', default=None)
    parser.add_argument('--s-min', type=float, default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--target-code', default=None)
    parser.add_argument('--impurity-code', default=None)
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--top-pairs', type=int, default=50,
                        help='Number of top pairs to refine in Stage 2')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-fill-thermal', action='store_true',
                        help='Disable auto-fill of missing thermal data')

    args = parser.parse_args()

    run_screening(
        target_smiles=args.target,
        impurity_smiles=args.impurity,
        mode=args.mode,
        config_path=args.config,
        s_min=args.s_min,
        output_dir=args.output_dir,
        target_code=args.target_code,
        impurity_code=args.impurity_code,
        top_n=args.top_n,
        top_pairs=args.top_pairs,
        no_plots=args.no_plots,
        verbose=args.verbose,
        fill_thermal=not args.no_fill_thermal,
    )
