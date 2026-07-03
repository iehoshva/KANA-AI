"""Configuration dataclass and preset variant definitions for KANA."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Config:
    SIGMA_DIM: int = 51
    SIGMA_CHANNELS: int = 1
    SCALAR_DIM: int = 10
    LATENT_A: int = 256
    LATENT_B: int = 256
    LATENT_Z: int = 512
    MAX_COMPONENTS: int = 10
    AGGREGATOR: str = 'molefrac_weighted'
    T_REF: float = 500.0
    R_GAS: float = 8.314
    GE_HEAD_HIDDEN: int = 1024
    LR: float = 1e-3
    LR_SCHEDULE: bool = True
    DROPOUT_RATE: float = 0.15
    SEED: int = 42
    W_LN_GAMMA: float = 1.0
    W_GAMMA_INF: float = 0.5
    W_HE: float = 0.3
    W_SE: float = 0.3
    W_BOUNDARY: float = 0.1


PRESETS: Dict[str, Config] = {
    '1': Config(LATENT_A=256, LATENT_B=256, LATENT_Z=512,  LR=1e-3),
    '2': Config(LATENT_A=512, LATENT_B=512, LATENT_Z=1024, LR=1e-3),
    '3': Config(LATENT_A=256, LATENT_B=256, LATENT_Z=512,  LR=1e-4),
    '4': Config(LATENT_A=512, LATENT_B=512, LATENT_Z=1024, LR=1e-4),
}


@dataclass
class PipelineConfig:
    """Runtime configuration for the post-training pipeline."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    checkpoint_dir: Path = field(default_factory=lambda: Path('best_cinn_checkpoint'))
    scalers_path: Path = field(default_factory=lambda: Path('scalers.pkl'))
    db_metadata: Path = field(default_factory=lambda: Path('compound_metadata.db'))
    db_properties: Path = field(default_factory=lambda: Path('compound_properties.db'))
    orca_bin: str = ''
    orca_nprocs: int = 4
    orca_maxcore: int = 1000
    orca_timeout_sec: int = 86400
    orca_max_retries: int = 3
    model_preset: str = '1'
    output_dir: Path = field(default_factory=lambda: Path('output'))
    T_min: float = 278.15
    T_max: float = 373.15
    T_step: float = 5.0
    S_min_abs: float = 10.0
    S_min_des: float = 50.0
    MAE_ln_gamma: float = 0.0749

    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """Load config from YAML file."""
        p = Path(path)
        if not p.exists():
            return cls()

        with open(p) as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        for key, val in data.items():
            if hasattr(cfg, key):
                fld = cls.__dataclass_fields__[key]
                if fld.type == Path:
                    val = Path(val)
                object.__setattr__(cfg, key, val)
        return cfg

    def resolve(self, *parts: str) -> Path:
        """Resolve path relative to base_dir."""
        return self.base_dir.joinpath(*parts)

    def ensure_dirs(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tie_lines').mkdir(exist_ok=True)
        (self.output_dir / 'binodal_curves').mkdir(exist_ok=True)
        (self.output_dir / 'loading_curves').mkdir(exist_ok=True)
        (self.output_dir / 'selectivity_plots').mkdir(exist_ok=True)


def load_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    """Load pipeline config from YAML or use defaults."""
    if yaml_path:
        return PipelineConfig.from_yaml(yaml_path)
    # Try default locations
    for candidate in ['config.yaml', 'config.yml']:
        p = Path(candidate)
        if p.exists():
            return PipelineConfig.from_yaml(str(p))
    return PipelineConfig()
