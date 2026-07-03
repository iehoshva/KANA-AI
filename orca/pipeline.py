"""Full ab initio pipeline: RDKit → 3D geometry → ORCA → Parse → DB."""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from kana.config import PipelineConfig
from database.metadata_db import MetadataDB
from .runner import ORCARunner
from .parser import ORCAParser


class AbInitioPipeline:
    """End-to-end pipeline for generating molecular features via DFT."""

    def __init__(self, pipe_cfg: PipelineConfig, meta_db: MetadataDB):
        self.pipe_cfg = pipe_cfg
        self.meta_db = meta_db
        self.runner = ORCARunner(pipe_cfg)
        self.parser = ORCAParser()

    def is_available(self) -> bool:
        """Check if ORCA is available for ab initio calculations."""
        return self.runner.is_available()

    def process(self, smiles: str, compound_code: str,
                compound_name: str = '', work_dir: Optional[Path] = None) -> bool:
        """Run full pipeline for a molecule.

        Args:
            smiles: canonical SMILES string
            compound_code: database code (e.g., "COMP_42")
            compound_name: human-readable name
            work_dir: directory for temp files (default: output/orca_tmp)

        Returns:
            True if successful and saved to DB
        """
        if work_dir is None:
            work_dir = self.pipe_cfg.resolve('output', 'orca_tmp')
        work_dir.mkdir(parents=True, exist_ok=True)

        if not compound_name:
            compound_name = compound_code

        try:
            # Step 1: RDKit canonicalization + 3D geometry
            mol_data = self._rdkit_geometry(smiles)
            if mol_data is None:
                return False

            atom_symbols, positions = mol_data

            # Step 2: Write XYZ
            xyz_path = work_dir / f"{compound_code}.xyz"
            self.runner.write_xyz(atom_symbols, positions, xyz_path)

            # Step 3: Write ORCA input
            inp_path = work_dir / f"{compound_code}.inp"
            self.runner.write_input(xyz_path, inp_path)

            # Step 4: Run ORCA
            success = self.runner.run(compound_code, work_dir,
                                       max_retries=self.pipe_cfg.orca_max_retries)
            if not success:
                return False

            # Step 5: Parse output
            out_path = work_dir / f"{compound_code}.out"
            cosmo_path = work_dir / f"{compound_code}.orcacosmo"

            result = self.parser.parse_all(out_path, cosmo_path)
            if result is None:
                return False

            sigma, features = result

            # Step 6: Save to DB
            self.meta_db.insert_compound(
                compound_code=compound_code,
                compound_name=compound_name,
                canonical_smiles=smiles,
                sigma_51=sigma,
                homo=features['HOMO'],
                lumo=features['LUMO'],
                dipole=features['dipole'],
                m0=features.get('M0', 0.0),
                m1=features.get('M1', 0.0),
                m2=features.get('M2', 0.0),
                m3=features.get('M3', 0.0),
                m4=features.get('M4', 0.0),
            )

            return True

        except Exception as e:
            print(f"Ab initio pipeline failed for {compound_code}: {e}")
            return False

    def _rdkit_geometry(self, smiles: str) -> Optional[Tuple[list, np.ndarray]]:
        """Generate 3D geometry using RDKit.

        Returns:
            (atom_symbols, positions) or None if failed
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                return None

            mol = Chem.AddHs(mol)

            # ETKDGv3 conformer generation
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                print(f"Conformer embedding failed for: {smiles}")
                return None

            # MMFF94 geometry pre-optimization
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

            conf = mol.GetConformer()
            atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol()
                           for i in range(mol.GetNumAtoms())]
            positions = conf.GetPositions()

            return atom_symbols, positions

        except ImportError:
            print("RDKit not available. Install rdkit-pypi or use conda.")
            return None
        except Exception as e:
            print(f"RDKit geometry generation failed: {e}")
            return None
