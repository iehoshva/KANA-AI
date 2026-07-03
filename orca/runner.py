"""ORCA job submission, monitoring, and retry logic."""

import subprocess
import time
import platform
from pathlib import Path
from typing import Optional

from kana.config import PipelineConfig


class ORCARunner:
    """Manages ORCA DFT job execution."""

    def __init__(self, pipe_cfg: PipelineConfig):
        self.pipe_cfg = pipe_cfg
        self.orca_bin = pipe_cfg.orca_bin
        self.nprocs = pipe_cfg.orca_nprocs
        self.maxcore = pipe_cfg.orca_maxcore
        self.timeout = pipe_cfg.orca_timeout_sec

        if not self.orca_bin:
            raise ValueError(
                "ORCA binary path not configured. "
                "Set 'orca_bin' in config.yaml or PipelineConfig."
            )

    def is_available(self) -> bool:
        """Check if ORCA binary exists and is executable."""
        p = Path(self.orca_bin)
        return p.exists() and p.is_file()

    def write_input(self, xyz_path: Path, inp_path: Path,
                    method: str = 'BP86', basis: str = 'def2-TZVPD',
                    solvent: str = 'Water', extra: str = ''):
        """Generate ORCA input file."""
        lines = [
            f"! {method} {basis} Opt TightSCF COSMO({solvent}) RI def2/J",
            f"%maxcore {self.maxcore}",
            f"%pal nprocs {self.nprocs} end",
            "%output PrintLevel Mini Print[P_Mulliken] 1 end",
        ]
        if extra:
            lines.append(extra)
        lines.append(f"* xyzfile 0 1 {xyz_path.name}")

        inp_path.write_text("\n".join(lines) + "\n")

    def run(self, job_name: str, work_dir: Path,
            retry: int = 0, max_retries: int = 3) -> bool:
        """Execute ORCA job and wait for completion.

        Args:
            job_name: base name (without extension)
            work_dir: directory containing .inp file
            retry: current retry attempt
            max_retries: maximum retries

        Returns:
            True if completed normally, False otherwise
        """
        inp_file = work_dir / f"{job_name}.inp"
        out_file = work_dir / f"{job_name}.out"

        if not inp_file.exists():
            raise FileNotFoundError(f"Input file not found: {inp_file}")

        # Platform-aware execution
        if platform.system() == 'Windows':
            cmd = [str(self.orca_bin), str(inp_file)]
            with open(out_file, 'w') as f:
                proc = subprocess.Popen(
                    cmd, stdout=f, stderr=subprocess.STDOUT,
                    cwd=str(work_dir),
                )
        else:
            cmd = f'"{self.orca_bin}" "{inp_file}" > "{out_file}" 2>&1'
            proc = subprocess.Popen(
                cmd, shell=True, cwd=str(work_dir),
                start_new_session=True,
            )

        # Wait for completion
        start = time.time()
        while time.time() - start < self.timeout:
            if proc.poll() is not None:
                break
            time.sleep(30)
        else:
            proc.kill()
            return False

        if proc.returncode != 0:
            if retry < max_retries:
                return self._retry_with_fallback(job_name, work_dir, retry, max_retries)
            return False

        return self._check_normal_termination(out_file)

    def _check_normal_termination(self, out_file: Path) -> bool:
        """Check if ORCA terminated normally."""
        try:
            with open(out_file) as f:
                lines = f.readlines()[-30:]
            return any("ORCA TERMINATED NORMALLY" in line for line in lines)
        except FileNotFoundError:
            return False

    def _retry_with_fallback(self, job_name: str, work_dir: Path,
                              retry: int, max_retries: int) -> bool:
        """Retry with progressively relaxed settings."""
        inp_file = work_dir / f"{job_name}.inp"
        original_text = inp_file.read_text()

        if retry == 1:
            # Add SCFConvForced + SlowConv
            modified = original_text.replace(
                "TightSCF", "TightSCF SCFConvForced SlowConv"
            )
        elif retry == 2:
            # Fall back to smaller basis set
            modified = original_text.replace("def2-TZVPD", "def2-SVP")
        else:
            return False

        inp_file.write_text(modified)
        return self.run(job_name, work_dir, retry + 1, max_retries)

    def write_xyz(self, atom_symbols: list, positions, xyz_path: Path):
        """Write XYZ coordinate file."""
        with open(xyz_path, 'w') as f:
            f.write(f"{len(atom_symbols)}\n\n")
            for sym, pos in zip(atom_symbols, positions):
                f.write(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}\n")
