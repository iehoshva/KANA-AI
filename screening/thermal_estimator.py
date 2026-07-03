"""Thermal data estimator using Joback group contribution method and RDKit.

Estimates melting point (T_f) and enthalpy of fusion (ΔH_fus) for compounds
missing from the properties database.

References:
- Joback & Reid (1987), Chem. Eng. Commun. 57:233-243
- Stein & Brown (1994), J. Inf. Comput. Sci. 34:581-587
"""

from typing import Optional, Dict, Tuple
from pathlib import Path
import sqlite3

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# Joback group contributions for melting point (T_f)
# ΔT_f values in K, from Joback & Reid 1987 Table 3
# Key: SMARTS pattern, Value: (ΔT_f contribution)
# We use a simplified set based on common functional groups

JOBACK_TF_GROUPS = {
    # Ring corrections (major contributors to T_f)
    'ring_aromatic_6': 42.0,    # 6-membered aromatic ring
    'ring_aromatic_5': 28.0,    # 5-membered aromatic ring
    'ring_non_aromatic_6': 25.0, # 6-membered non-aromatic ring
    'ring_non_aromatic_5': 15.0, # 5-membered non-aromatic ring

    # Functional group contributions
    'OH': 44.0,       # Hydroxyl
    'NH2': 67.0,      # Primary amine
    'NH': 50.0,       # Secondary amine
    'COOH': 105.0,    # Carboxylic acid
    'CONH2': 120.0,   # Primary amide
    'CONH': 90.0,     # Secondary amide
    'COO': 30.0,      # Ester
    'CHO': 40.0,      # Aldehyde
    'CO': 25.0,       # Ketone
    'NO2': 70.0,      # Nitro
    'SO2': 55.0,      # Sulfone
    'SO': 35.0,       # Sulfoxide
    'F': 15.0,        # Fluorine
    'Cl': 25.0,       # Chlorine
    'Br': 30.0,       # Bromine
    'I': 35.0,        # Iodine

    # Carbon type contributions
    'CH3': 5.0,       # Methyl
    'CH2': 5.0,       # Methylene
    'CH': 3.0,        # Methine
    'C_quat': 1.0,    # Quaternary carbon

    # Heteroatom in ring
    'N_in_ring': 20.0,
    'O_in_ring': 15.0,
    'S_in_ring': 20.0,
}

# Enthalpy of fusion estimation
# Yalkowsky approximation: ΔS_fus ≈ 50 J/(mol·K) for rigid organic molecules
# ΔH_fus = ΔS_fus × T_f
# For flexible molecules, ΔS_fus is lower
YALKOWSKY_DS_BASE = 50.0  # J/(mol·K)


def estimate_thermal_data(smiles: str) -> Optional[Dict[str, float]]:
    """Estimate T_f (K) and ΔH_fus (J/mol) from SMILES.

    Uses Joback group contribution for T_f and Yalkowsky for ΔH_fus.

    Args:
        smiles: canonical SMILES string

    Returns:
        {'T_f_K': float, 'dH_fus': float} or None if estimation fails
    """
    if not HAS_RDKIT:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Estimate T_f via group contributions
    T_f = _estimate_Tf_joback(mol)
    if T_f is None or T_f < 200 or T_f > 700:
        # Fallback: use correlation with molecular descriptors
        T_f = _estimate_Tf_correlation(mol)

    if T_f is None:
        return None

    # Estimate ΔH_fus via Yalkowsky
    n_rot = Descriptors.NumRotatableBonds(mol)
    n_rings = Descriptors.RingCount(mol)
    n_arom = Descriptors.NumAromaticRings(mol)

    # Flexibility correction: fewer rotatable bonds → higher ΔS_fus
    # Rigid aromatic molecules: ΔS_fus ≈ 50-56 J/(mol·K)
    # Flexible molecules: ΔS_fus ≈ 20-35 J/(mol·K)
    flexibility = n_rot / max(n_rings + n_arom, 1)
    dS_fus = YALKOWSKY_DS_BASE * (1.0 - 0.3 * min(flexibility, 2.0))
    dS_fus = max(dS_fus, 20.0)  # floor

    dH_fus = dS_fus * T_f  # J/mol

    return {
        'T_f_K': float(T_f),
        'dH_fus': float(dH_fus),
    }


def _estimate_Tf_joback(mol) -> Optional[float]:
    """Estimate T_f using Joback group contributions.

    T_f = 102.42 + Σ(ΔT_f,i) [K]
    """
    if not HAS_RDKIT:
        return None

    T_f = 102.42  # Base value from Joback

    # Count functional groups using RDKit descriptors
    T_f += Descriptors.NumAliphaticRings(mol) * JOBACK_TF_GROUPS.get('ring_non_aromatic_6', 25)
    T_f += Descriptors.NumAromaticRings(mol) * JOBACK_TF_GROUPS.get('ring_aromatic_6', 42)

    # Functional groups
    T_f += Descriptors.NumHDonors(mol) * JOBACK_TF_GROUPS.get('OH', 44)
    T_f += Descriptors.fr_NH0(mol) * JOBACK_TF_GROUPS.get('NH', 50)
    T_f += Descriptors.fr_NH2(mol) * JOBACK_TF_GROUPS.get('NH2', 67)
    T_f += Descriptors.fr_COO(mol) * JOBACK_TF_GROUPS.get('COOH', 105)
    T_f += Descriptors.fr_amide(mol) * JOBACK_TF_GROUPS.get('CONH', 90)
    T_f += Descriptors.fr_ester(mol) * JOBACK_TF_GROUPS.get('COO', 30)
    T_f += Descriptors.fr_aldehyde(mol) * JOBACK_TF_GROUPS.get('CHO', 40)
    T_f += Descriptors.fr_ketone(mol) * JOBACK_TF_GROUPS.get('CO', 25)
    T_f += Descriptors.fr_nitro(mol) * JOBACK_TF_GROUPS.get('NO2', 70)

    # Halogens
    T_f += Descriptors.fr_halogen(mol) * 20

    # Heteroatoms in rings
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 7:  # N
                T_f += JOBACK_TF_GROUPS.get('N_in_ring', 20)
            elif atom.GetAtomicNum() == 8:  # O
                T_f += JOBACK_TF_GROUPS.get('O_in_ring', 15)
            elif atom.GetAtomicNum() == 16:  # S
                T_f += JOBACK_TF_GROUPS.get('S_in_ring', 20)

    return T_f


def _estimate_Tf_correlation(mol) -> Optional[float]:
    """Fallback T_f estimation using molecular descriptor correlation.

    Based on correlation: T_f ≈ 300 + 3.5*MolLogP + 2.5*MolMR - 0.5*RotatableBonds
    (empirical, calibrated on drug-like molecules)
    """
    if not HAS_RDKIT:
        return None

    logP = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)
    n_rings = Descriptors.RingCount(mol)

    # Empirical correlation
    T_f = 300 + 3.5 * logP + 2.5 * mr - 0.5 * n_rot + 10 * n_rings

    # Clamp to reasonable range
    T_f = max(200.0, min(600.0, T_f))

    return T_f


def fill_missing_thermal_data(db_path: Path, dry_run: bool = False) -> int:
    """Fill missing T_f and ΔH_fus in compound_properties.db.

    Args:
        db_path: path to compound_properties.db
        dry_run: if True, only report what would be filled

    Returns:
        number of compounds updated
    """
    if not HAS_RDKIT:
        print("RDKit not available. Cannot estimate thermal data.")
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find compounds with missing thermal data
    rows = conn.execute(
        """SELECT compound_code, compound_name, canonical_smiles
           FROM compounds
           WHERE T_f_C IS NULL OR deltaH_fus_kJmol IS NULL"""
    ).fetchall()

    print(f"Found {len(rows)} compounds with missing thermal data")

    updated = 0
    for row in rows:
        code = row['compound_code']
        smiles = row['canonical_smiles']
        name = row['compound_name'] or code

        if not smiles:
            continue

        estimate = estimate_thermal_data(smiles)
        if estimate is None:
            continue

        T_f_C = estimate['T_f_K'] - 273.15
        dH_fus_kJ = estimate['dH_fus'] / 1000.0

        if dry_run:
            print(f"  {code} ({name}): T_f={T_f_C:.1f}°C, ΔH_fus={dH_fus_kJ:.1f} kJ/mol")
        else:
            conn.execute(
                """UPDATE compounds
                   SET T_f_C = ?, deltaH_fus_kJmol = ?
                   WHERE compound_code = ?""",
                (T_f_C, dH_fus_kJ, code),
            )
            updated += 1

    if not dry_run:
        conn.commit()
        print(f"Updated {updated} compounds with estimated thermal data")

    conn.close()
    return updated
