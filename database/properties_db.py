"""DB #2: compound_properties.db — OUTPUT database (names, IUPAC, thermal data)."""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np


class PropertiesDB:
    """Interface to compound_properties.db (DB #2: OUTPUT)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Properties DB not found: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_compound_name(self, compound_code: str) -> str:
        """Get human-readable compound name. Falls back to code if not found."""
        row = self.conn.execute(
            "SELECT compound_name FROM compounds WHERE compound_code=?",
            (compound_code,),
        ).fetchone()
        return row['compound_name'] if row else compound_code

    def get_iupac_name(self, compound_code: str) -> Optional[str]:
        """Get IUPAC systematic name."""
        row = self.conn.execute(
            "SELECT iupac_name FROM compounds WHERE compound_code=?",
            (compound_code,),
        ).fetchone()
        return row['iupac_name'] if row else None

    def get_thermal_data(self, compound_code: str) -> Optional[Dict[str, float]]:
        """Get melting point and enthalpy of fusion.

        Returns:
            {'T_f_K': float, 'dH_fus': float} in K and J/mol, or None
        """
        row = self.conn.execute(
            "SELECT T_f_C, deltaH_fus_kJmol FROM compounds WHERE compound_code=?",
            (compound_code,),
        ).fetchone()

        if row is None or row['T_f_C'] is None:
            return None

        return {
            'T_f_K': float(row['T_f_C']) + 273.15,
            'dH_fus': float(row['deltaH_fus_kJmol']) * 1000.0,
        }

    def get_thermal_pair(self, code_a: str, code_b: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Fetch thermal data for two compounds.

        Returns:
            {code_a: {'T_f_K': ..., 'dH_fus': ...}, code_b: ...} or None if incomplete
        """
        rows = self.conn.execute(
            """SELECT compound_code, T_f_C, deltaH_fus_kJmol
               FROM compounds WHERE compound_code IN (?, ?)""",
            (code_a, code_b),
        ).fetchall()

        data = {}
        for row in rows:
            if row['T_f_C'] is not None and row['deltaH_fus_kJmol'] is not None:
                data[row['compound_code']] = {
                    'T_f_K': float(row['T_f_C']) + 273.15,
                    'dH_fus': float(row['deltaH_fus_kJmol']) * 1000.0,
                }

        if len(data) < 2:
            return None

        return data

    def get_smiles(self, compound_code: str) -> Optional[str]:
        """Get canonical SMILES."""
        row = self.conn.execute(
            "SELECT canonical_smiles FROM compounds WHERE compound_code=?",
            (compound_code,),
        ).fetchone()
        return row['canonical_smiles'] if row else None

    def get_all_names(self) -> Dict[str, str]:
        """Get mapping of compound_code -> compound_name for all entries."""
        rows = self.conn.execute(
            "SELECT compound_code, compound_name FROM compounds"
        ).fetchall()
        return {r['compound_code']: r['compound_name'] for r in rows}

    def get_all_codes(self) -> List[str]:
        """Get all compound codes."""
        rows = self.conn.execute("SELECT compound_code FROM compounds").fetchall()
        return [r['compound_code'] for r in rows]
