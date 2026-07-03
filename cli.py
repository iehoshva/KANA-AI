"""CLI entry point for KANA extraction screening."""

import argparse
import sys
from pathlib import Path

# Ensure package imports work when run as script
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        prog='kana-screen',
        description='KANA: Kimia-informed Artificial Neural-network coffenovA — Extraction Screening',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DES mode screening
  kana-screen --target "OC(=O)C1=CC(=C(O)C(O)=C1)OC1OC(CO)C(O)C(O)C1O" \\
              --impurity "CN1C=NC2=C1C(=O)N(C(=O)N2)C" \\
              --mode DES

  # ABS mode with custom selectivity threshold
  kana-screen --target "OC(=O)C1=CC(=C(O)C(O)=C1)OC1OC(CO)C(O)C(O)C1O" \\
              --impurity "CN1C=NC2=C1C(=O)N(C(=O)N2)C" \\
              --mode ABS --s-min 15

  # Custom config file
  kana-screen --target "SMILES" --impurity "SMILES" \\
              --config my_config.yaml --mode DES
        """,
    )

    parser.add_argument('--target', required=True,
                        help='SMILES of target compound (bioactive to extract)')
    parser.add_argument('--impurity', required=True,
                        help='SMILES of impurity compound (to separate)')
    parser.add_argument('--mode', choices=['ABS', 'DES'], default='DES',
                        help='Screening mode: ABS (aqueous biphasic) or DES (deep eutectic)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: auto-detect)')
    parser.add_argument('--s-min', type=float, default=None,
                        help='Minimum selectivity threshold (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--target-code', type=str, default=None,
                        help='Compound code for target in DB (if known)')
    parser.add_argument('--impurity-code', type=str, default=None,
                        help='Compound code for impurity in DB (if known)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top systems to show in report (default: 10)')
    parser.add_argument('--top-pairs', type=int, default=50,
                        help='Number of top pairs to refine in Stage 2 (default: 50)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--no-fill-thermal', action='store_true',
                        help='Disable auto-fill of missing thermal data')

    args = parser.parse_args()

    # Import here to avoid slow startup on --help
    from run_screening import run_screening

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


if __name__ == '__main__':
    main()
