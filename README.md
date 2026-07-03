# KANA-AI: Post-Training Extraction Screening Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **KANA** — Kimia-informed Artificial Neural-network coffenovA
> Solvent screening for liquid-liquid extraction using hard-constrained thermodynamic neural networks.

---

## What is KANA-AI?

KANA-AI screens solvent systems for separating target compounds from impurities. Given SMILES inputs, it evaluates all possible solvent pairs from a 160-compound database, ranks them by selectivity, and provides thermodynamic validation.

**Two modes:**
- **DES** — Deep Eutectic Solvent (solid-liquid extraction)
- **ABS** — Aqueous Biphasic System (liquid-liquid extraction)

**Key innovation:** The model enforces Gibbs-Duhem, Gibbs-Helmholtz, and pure-component boundary constraints exactly — predictions are thermodynamically consistent by construction.

---

## Performance

| Metric | Value |
|--------|-------|
| Screening time | ~100 seconds (160 compounds, laptop CPU) |
| Validation MAE | 0.0749 ln γ units (epoch 940) |
| Architecture | HardConstrainedCINN (LATENT=256) |
| Permutation invariant | ✓ (any compound can be any component) |

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/KANA-AI.git
cd KANA-AI

# Create conda environment (or use existing)
conda create -n KANA python=3.10
conda activate KANA

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data Files

Place these files in the **parent directory** (`../`):

| File | Description | Size |
|------|-------------|------|
| `compound_metadata.db` | 160 compounds: sigma profiles + quantum features | ~90 KB |
| `compound_properties.db` | Compound names + thermal data | ~30 KB |
| `scalers.pkl` | Pre-trained StandardScaler objects | ~2 KB |
| `best_cinn_checkpoint/` | Flax checkpoint (epoch 940) | ~5.5 MB |

### 3. Run Screening

```bash
# DES mode: screen for deep eutectic solvent systems
python cli.py \
  --target "OC(=O)C1=CC(=C(O)C(O)=C1)OC1OC(CO)C(O)C(O)C1O" \
  --impurity "CN1C=NC2=C1C(=O)N(C(=O)N2)C" \
  --mode DES

# ABS mode: screen for aqueous biphasic systems
python cli.py \
  --target "OC(=O)C1=CC(=C(O)C(O)=C1)OC1OC(CO)C(O)C(O)C1O" \
  --impurity "CN1C=NC2=C1C(=O)N(C(=O)N2)C" \
  --mode ABS
```

### 4. Check Results

```bash
# Ranked solvent systems
cat output/extraction_screening_results.csv

# Human-readable report
cat output/screening_report.md

# Plots (if matplotlib installed)
ls output/selectivity_plots/
```

---

## How It Works

### Two-Stage Fast Screening

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Coarse Scan (~70s)                                │
│    • 152×152 = 23,104 solvent pairs                         │
│    • 1 temperature × 1 ratio per pair                       │
│    • Single batched JAX forward pass per chunk              │
│    → Ranks all pairs by selectivity                         │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Full Grid (~30s)                                  │
│    • Top-50 pairs from Stage 1                              │
│    • 14 temperatures × 13 ratios = 182 points per pair      │
│    • Full uncertainty quantification                        │
│    → Detailed results with confidence intervals             │
└─────────────────────────────────────────────────────────────┘
```

### Thermodynamic Validation

| Check | DES Mode | ABS Mode |
|-------|----------|----------|
| Eutectic depression (ΔT_f > 0) | ✓ | — |
| Phase stability (Michelsen TPD) | — | ✓ |
| Isoactivity solver | — | ✓ |
| Gibbs mixing concavity | — | ✓ |
| Selectivity threshold | ✓ | ✓ |
| Capacity threshold | ✓ | ✓ |
| Confidence interval | ✓ | ✓ |

### Thermal Data Auto-Fill

Missing melting points and enthalpies of fusion are estimated using:
- **Joback group contribution** for T_f
- **Yalkowsky approximation** for ΔH_fus

Runs automatically before screening. Disable with `--no-fill-thermal`.

---

## CLI Reference

```
usage: python cli.py [options]

Required:
  --target SMILES        Target compound SMILES
  --impurity SMILES      Impurity compound SMILES

Optional:
  --mode {ABS,DES}       Screening mode (default: DES)
  --config PATH          Path to config.yaml
  --s-min FLOAT          Minimum selectivity threshold
  --output-dir DIR       Output directory
  --target-code CODE     Compound code for target in DB
  --impurity-code CODE   Compound code for impurity in DB
  --top-n N              Top systems in report (default: 10)
  --top-pairs N          Pairs to refine in Stage 2 (default: 50)
  --no-plots             Skip plot generation
  --no-fill-thermal      Skip thermal data auto-fill
  --verbose, -v          Verbose output
```

---

## Configuration

Edit `config.yaml`:

```yaml
# Model checkpoint (relative to base_dir)
checkpoint_dir: "best_cinn_checkpoint"

# Database paths (relative to base_dir)
db_metadata: "compound_metadata.db"
db_properties: "compound_properties.db"

# ORCA DFT binary (for ab initio calculations)
# Leave empty to disable
orca_bin: ""

# Temperature sweep range (Kelvin)
T_min: 278.15    # 5°C
T_max: 373.15    # 100°C
T_step: 5.0      # 5°C steps

# Selectivity thresholds
S_min_des: 50.0
S_min_abs: 10.0

# Model validation MAE (ln gamma units)
MAE_ln_gamma: 0.0749
```

---

## Output Format

### CSV Columns

| Column | Description |
|--------|-------------|
| `solvent_A`, `solvent_B` | Human-readable compound names |
| `solvent_A_code`, `solvent_B_code` | Database codes (COMP_X) |
| `T_opt_K` | Optimal temperature (K) |
| `ratio_A`, `ratio_B` | Molar ratio |
| `S_inf` | Infinite dilution selectivity |
| `S_inf_lower95`, `S_inf_upper95` | 95% confidence interval |
| `capacity_inf` | Extraction capacity |
| `confidence` | HIGH / MEDIUM / LOW |
| `des_valid` | DES eutectic validation passed |
| `lle_valid` | LLE phase split validation passed |

### Markdown Report

Generated automatically with:
- Top-10 recommended systems
- DES/LLE validation summary
- Model details

---

## Adding New Compounds

If a compound is not in the database:

1. Install ORCA (https://orcaforum.kofo.mpg.de/)
2. Set `orca_bin` in `config.yaml`
3. Run screening — missing compounds trigger DFT automatically
4. Results saved to `compound_metadata.db` for future use

**ORCA input settings:**
```
! BP86 def2-TZVPD Opt TightSCF COSMO(Water) RI def2/J
%maxcore 1000
%pal nprocs 4 end
```

---

## Architecture

```
Component i
    +-- Sigma Profile (51-bin)  →  SigmaEncoder  ──┐
    +-- Scalar Features (10-dim) → ScalarEncoder ──┤
                                                   ↓
                                     z_i = [z_a || z_b]  (512-dim)
                                                   ↓
                              ┌─────────────────────────────┐
                              │  MoleFractionWeighted        │
                              │  Aggregator                  │
                              │  z_mix = Σ x_i · z_i        │
                              └─────────────────────────────┘
                                                   ↓
                                    GE_PredictionHead
                              gE/RT = f(z_mix, T) · Σ_{i<j} x_i x_j
                                                   ↓
                              ThermodynamicEngine (JAX autodiff)
                              +-- ln γ_i = ∂(n·gE/RT)/∂n_i
                              +-- h^E = -T² · ∂(gE/T)/∂T
                              +-- s^E = (h^E - gE)/T
```

**Hard constraints:**
- Gibbs-Duhem: exact via autodiff
- Pure-component boundary: via Σ_{i<j} x_i x_j mask
- Gibbs-Helmholtz: via temperature derivatives
- Permutation invariance: via mole-fraction weighted aggregation

---

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | ✓ Full | Recommended for production |
| Windows | ✓ Supported | Use forward slashes in paths |
| macOS | ✓ CPU | GPU via Metal plugin |

**No internet required** at runtime. All data bundled locally.

---

## Dependencies

```
jax>=0.4.0
jaxlib>=0.4.0
flax>=0.8.0
optax>=0.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.10.0
pyyaml>=6.0
```

Optional: `rdkit` (for ab initio calculations and thermal data estimation)

---

## License

GNU General Public License v3.0 (GPLv3). See [LICENSE](LICENSE).

---

## Acknowledgements

Built on the [KANA](https://github.com/Flowychie/KANA) pre-training framework. Uses JAX/Flax for differentiable scientific computing.
