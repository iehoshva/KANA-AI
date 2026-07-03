"""Parse ORCA output files (.out and .orcacosmo)."""

import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


class ORCAParser:
    """Parses ORCA output files to extract quantum features and sigma profiles."""

    @staticmethod
    def parse_output(out_path: Path) -> dict:
        """Extract quantum features from ORCA .out file.

        Returns:
            dict with keys: HOMO, LUMO, gap, dipole, M0, M1, M2, M3, M4
        """
        with open(out_path) as f:
            text = f.read()

        result = {}

        # HOMO energy
        m = re.search(r'E\(HOMO\).*?:\s*(-?\d+\.\d+)', text)
        if m:
            result['HOMO'] = float(m.group(1))
        else:
            # Try alternative pattern
            m = re.search(r'ORBITAL ENERGIES.*?(\d+)\s+(-?\d+\.\d+)\s+', text, re.DOTALL)
            if m:
                result['HOMO'] = float(m.group(2))

        # LUMO energy
        m = re.search(r'E\(LUMO\).*?:\s*(-?\d+\.\d+)', text)
        if m:
            result['LUMO'] = float(m.group(1))

        # HOMO-LUMO gap
        if 'HOMO' in result and 'LUMO' in result:
            result['gap'] = result['HOMO'] - result['LUMO']
        else:
            m = re.search(r'HOMO-LUMO Gap.*?:\s*(-?\d+\.\d+)', text)
            if m:
                result['gap'] = float(m.group(1))

        # Dipole moment
        m = re.search(r'Total Dipole Moment.*?:\s*(\d+\.\d+)', text)
        if m:
            result['dipole'] = float(m.group(1))
        else:
            m = re.search(r'Total Dipole Moment\s+:\s+(\d+\.\d+)', text)
            if m:
                result['dipole'] = float(m.group(1))

        # Mulliken charges
        charges = ORCAParser._parse_mulliken_charges(text)
        if charges:
            result['M0'] = max(charges) - min(charges)  # charge range
            result['M1'] = np.std(charges)  # charge std
            result['M2'] = 0.0  # secondary metric (placeholder)
            result['M3'] = 0.0  # padding
            result['M4'] = 0.0  # padding
        else:
            result.setdefault('M0', 0.0)
            result.setdefault('M1', 0.0)
            result.setdefault('M2', 0.0)
            result.setdefault('M3', 0.0)
            result.setdefault('M4', 0.0)

        return result

    @staticmethod
    def _parse_mulliken_charges(text: str) -> Optional[list]:
        """Extract Mulliken atomic charges from ORCA output."""
        # Pattern: MULLIKEN ATOMIC CHARGES block
        pattern = r'MULLIKEN ATOMIC CHARGES\s*\n\s*-+\s*\n(.*?)(?:\n\s*-+|\n\s*\n)'
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return None

        charges = []
        for line in m.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    charges.append(float(parts[-1]))
                except ValueError:
                    continue
        return charges if charges else None

    @staticmethod
    def parse_sigma_profile(cosmofile_path: Path) -> np.ndarray:
        """Extract 51-bin sigma profile from .orcacosmo file.

        Returns:
            np.ndarray of shape (51,) with sigma profile values
        """
        sigma = np.zeros(51, dtype='float32')

        if not cosmofile_path.exists():
            return sigma

        with open(cosmofile_path) as f:
            lines = f.readlines()

        # Parse sigma profile: look for the histogram data
        # Format varies by ORCA version, handle both
        idx = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('$'):
                continue
            parts = line.split()
            if len(parts) >= 2 and idx < 51:
                try:
                    sigma[idx] = float(parts[-1])
                    idx += 1
                except ValueError:
                    continue

        return sigma

    @staticmethod
    def check_termination(out_path: Path) -> bool:
        """Check if ORCA terminated normally."""
        try:
            with open(out_path) as f:
                lines = f.readlines()[-30:]
            return any("ORCA TERMINATED NORMALLY" in line for line in lines)
        except FileNotFoundError:
            return False

    @staticmethod
    def parse_all(out_path: Path, cosmo_path: Path) -> Optional[Tuple[np.ndarray, dict]]:
        """Parse both .out and .orcacosmo files.

        Returns:
            (sigma_51, quantum_features_dict) or None if failed
        """
        if not ORCAParser.check_termination(out_path):
            return None

        features = ORCAParser.parse_output(out_path)
        sigma = ORCAParser.parse_sigma_profile(cosmo_path)

        required = ['HOMO', 'LUMO', 'dipole']
        if not all(k in features for k in required):
            return None

        return sigma, features
