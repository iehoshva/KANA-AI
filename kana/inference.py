"""Post-training inference wrapper: checkpoint loading, scaler application, batched forward pass."""

import pickle
from pathlib import Path
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from .config import Config, PRESETS, PipelineConfig
from .architecture import HardConstrainedCINN
from .thermodynamics import ThermodynamicEngine


class KANAInference:
    """Inference engine for KANA post-training screening."""

    def __init__(self, pipeline_cfg: PipelineConfig):
        self.pipe_cfg = pipeline_cfg
        self.cfg: Config = PRESETS[pipeline_cfg.model_preset]
        self.model = HardConstrainedCINN(cfg=self.cfg)
        self.engine = ThermodynamicEngine(self.cfg)
        self.params = None
        self.scaler_scalar = None
        self.scaler_sigma = None

    def load(self):
        """Load checkpoint, scalers, and initialize model."""
        self._load_scalers()
        self._load_checkpoint()
        self._warmup()

    def _load_scalers(self):
        scalers_path = self.pipe_cfg.resolve(self.pipe_cfg.scalers_path).resolve()
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers not found: {scalers_path}")

        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)

        self.scaler_scalar = scalers['scaler_scalar']
        self.scaler_sigma = scalers['scaler_sigma']

        assert self.scaler_scalar.n_features_in_ == self.cfg.SCALAR_DIM, \
            f"Scaler expects {self.scaler_scalar.n_features_in_} scalar features, config has {self.cfg.SCALAR_DIM}"
        assert self.scaler_sigma.n_features_in_ == self.cfg.SIGMA_DIM, \
            f"Scaler expects {self.scaler_sigma.n_features_in_} sigma features, config has {self.cfg.SIGMA_DIM}"

    def _load_checkpoint(self):
        import optax
        from flax.training import train_state

        ckpt_dir = self.pipe_cfg.resolve(self.pipe_cfg.checkpoint_dir).resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        # First init the model to get a target pytree with correct structure
        dummy_sigmas = jnp.ones((1, self.cfg.MAX_COMPONENTS, self.cfg.SIGMA_DIM, 1))
        dummy_scalars = jnp.ones((1, self.cfg.MAX_COMPONENTS, self.cfg.SCALAR_DIM))
        dummy_mask = jnp.ones((1, self.cfg.MAX_COMPONENTS), dtype=bool)
        dummy_T = jnp.array([298.15])
        dummy_x = jnp.ones((1, self.cfg.MAX_COMPONENTS)) / self.cfg.MAX_COMPONENTS
        rng = jax.random.PRNGKey(0)
        variables = self.model.init(rng, dummy_sigmas, dummy_scalars, dummy_mask, dummy_T, dummy_x, training=False)

        # Create a dummy TrainState as target (matching what was saved)
        tx = optax.adam(self.cfg.LR)
        dummy_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=tx,
        )

        # Restore using flax checkpoints with the full TrainState target
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=dummy_state,
        )
        self.params = restored_state.params

    def _warmup(self):
        """Warmup JIT compilation with a dummy forward pass."""
        dummy_sigma = jnp.ones((1, self.cfg.MAX_COMPONENTS, self.cfg.SIGMA_DIM, 1))
        dummy_scalar = jnp.ones((1, self.cfg.MAX_COMPONENTS, self.cfg.SCALAR_DIM))
        dummy_mask = jnp.ones((1, self.cfg.MAX_COMPONENTS), dtype=bool)
        dummy_T = jnp.array([298.15])
        dummy_x = jnp.ones((1, self.cfg.MAX_COMPONENTS)) / self.cfg.MAX_COMPONENTS

        _ = self.model.apply(
            {'params': self.params},
            dummy_sigma, dummy_scalar, dummy_mask, dummy_T, dummy_x,
            training=False,
        )

    def scale_features(self, sigma_raw: np.ndarray, scalar_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply pre-trained scalers to raw features.

        Args:
            sigma_raw: shape (n_comp, 51) raw sigma profiles
            scalar_raw: shape (n_comp, 8) raw quantum features

        Returns:
            scaled_sigma: shape (n_comp, 51, 1)
            scaled_scalar: shape (n_comp, 8)
        """
        sigma_2d = sigma_raw.reshape(-1, self.cfg.SIGMA_DIM)
        scalar_2d = scalar_raw.reshape(-1, self.cfg.SCALAR_DIM)

        sigma_scaled = self.scaler_sigma.transform(sigma_2d)
        scalar_scaled = self.scaler_scalar.transform(scalar_2d)

        return sigma_scaled.reshape(-1, self.cfg.SIGMA_DIM, 1), scalar_scaled

    def predict_gE_RT(self, sigma_batch, scalar_batch, mask_batch, T_batch, x_batch):
        """Batched forward pass returning gE/RT."""
        return self.model.apply(
            {'params': self.params},
            sigma_batch, scalar_batch, mask_batch, T_batch, x_batch,
            training=False,
        )

    def predict_ln_gamma(self, sigma_batch, scalar_batch, mask_batch, T_batch, n_batch):
        """Batched ln gamma computation via autodiff."""
        return self.engine.compute_ln_gamma(
            self.params, self.model.apply,
            sigma_batch, scalar_batch, mask_batch, T_batch, n_batch,
        )

    def predict_hE(self, sigma_batch, scalar_batch, mask_batch, T_batch, x_batch):
        """Batched excess enthalpy."""
        return self.engine.compute_hE(
            self.params, self.model.apply,
            sigma_batch, scalar_batch, mask_batch, T_batch, x_batch,
        )

    def predict_sE(self, sigma_batch, scalar_batch, mask_batch, T_batch, x_batch, n_batch):
        """Batched excess entropy."""
        return self.engine.compute_sE(
            self.params, self.model.apply,
            sigma_batch, scalar_batch, mask_batch, T_batch, x_batch, n_batch,
        )
