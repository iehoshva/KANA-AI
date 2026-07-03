"""DB #1: compound_metadata.db — ENGINE database (SMILES, sigma profiles, quantum features)."""

import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np


class MetadataDB:
    """Interface to compound_metadata.db (DB #1: ENGINE)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Metadata DB not found: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_features(self, compound_code: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fetch sigma profile and quantum features for a compound.

        Returns:
            (sigma_51, scalars_10) or None if not found / incomplete
            scalars_10: [HOMO, LUMO, Dipole, M0, M1, M2, M3, M4, 0, 0]
        """
        row = self.conn.execute(
            """SELECT sigma_profile, HOMO, LUMO, Dipole, M0, M1, M2, M3, M4
               FROM compounds WHERE compound_code=?""",
            (compound_code,),
        ).fetchone()

        if row is None or row['sigma_profile'] is None:
            return None

        sigma = np.fromstring(row['sigma_profile'], sep=',', dtype='float32')
        if sigma.shape[0] != 51:
            return None

        # 8 quantum features + 2 padding zeros = 10 total (matching scaler)
        scalars = np.array([
            row['HOMO'] or 0.0, row['LUMO'] or 0.0, row['Dipole'] or 0.0,
            row['M0'] or 0.0, row['M1'] or 0.0, row['M2'] or 0.0,
            row['M3'] or 0.0, row['M4'] or 0.0,
            0.0, 0.0,
        ], dtype='float32')

        if np.any(np.isnan(scalars[:3])):
            return None

        return sigma, scalars

    def get_smiles(self, compound_code: str) -> Optional[str]:
        """Get canonical SMILES for a compound."""
        row = self.conn.execute(
            "SELECT canonical_smiles FROM compounds WHERE compound_code=?",
            (compound_code,),
        ).fetchone()
        return row['canonical_smiles'] if row else None

    def get_compound_name(self, compound_code: str) -> Optional[str]:
        """Get compound name (from metadata db)."""
        row = self.conn.execute(
            "SELECT compound_name FROM compounds WHERE compound_code=?",
            (compound_code,),
        ).fetchone()
        return row['compound_name'] if row else None

    def get_all_with_features(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get all compounds that have complete sigma + quantum features."""
        rows = self.conn.execute(
            """SELECT compound_code, sigma_profile,
                      HOMO, LUMO, Dipole, M0, M1, M2, M3, M4
               FROM compounds
               WHERE sigma_profile IS NOT NULL
                 AND HOMO IS NOT NULL AND LUMO IS NOT NULL
                 AND Dipole IS NOT NULL"""
        ).fetchall()

        result = {}
        for row in rows:
            sigma = np.fromstring(row['sigma_profile'], sep=',', dtype='float32')
            if sigma.shape[0] != 51:
                continue
            # 8 quantum features + 2 padding zeros = 10 total
            scalars = np.array([
                row['HOMO'] or 0.0, row['LUMO'] or 0.0, row['Dipole'] or 0.0,
                row['M0'] or 0.0, row['M1'] or 0.0, row['M2'] or 0.0,
                row['M3'] or 0.0, row['M4'] or 0.0,
                0.0, 0.0,
            ], dtype='float32')
            if not np.any(np.isnan(scalars[:3])):
                result[row['compound_code']] = (sigma, scalars)

        return result

    def get_all_codes(self) -> List[str]:
        """Get all compound codes in the database."""
        rows = self.conn.execute("SELECT compound_code FROM compounds").fetchall()
        return [r['compound_code'] for r in rows]

    def insert_compound(self, compound_code: str, compound_name: str,
                        canonical_smiles: str, sigma_51: np.ndarray,
                        homo: float, lumo: float, dipole: float,
                        m0: float, m1: float, m2: float,
                        m3: float = 0.0, m4: float = 0.0):
        """Insert or update a compound in the metadata database."""
        sigma_csv = ",".join(f"{x:.6f}" for x in sigma_51)
        self.conn.execute(
            """INSERT OR REPLACE INTO compounds
               (compound_code, compound_name, canonical_smiles,
                sigma_profile, HOMO, LUMO, Dipole, M0, M1, M2, M3, M4)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (compound_code, compound_name, canonical_smiles,
             sigma_csv, homo, lumo, dipole, m0, m1, m2, m3, m4),
        )
        self.conn.commit()

    def has_compound(self, compound_code: str) -> bool:
        """Check if compound exists with complete features."""
        return self.get_features(compound_code) is not None
