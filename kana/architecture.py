"""Flax Linen modules: encoders, aggregator, prediction head, full model."""

import jax.numpy as jnp
import flax.linen as nn

from .config import Config


class SigmaEncoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(features=32, kernel_size=(5,), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3,), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.cfg.LATENT_A)(x)
        x = nn.relu(x)
        return x


class ScalarEncoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=not training)(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=not training)(x)
        x = nn.Dense(self.cfg.LATENT_B)(x)
        x = nn.relu(x)
        return x


class ComponentEncoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, sigma, scalar, training=True):
        z_a = SigmaEncoder(cfg=self.cfg)(sigma, training=training)
        z_b = ScalarEncoder(cfg=self.cfg)(scalar, training=training)
        return jnp.concatenate([z_a, z_b], axis=-1)


class MoleFractionWeightedAggregator(nn.Module):
    cfg: Config
    output_dim: int = 128

    @nn.compact
    def __call__(self, z_components, x, mask):
        mask_f = mask.astype(jnp.float32)
        x_masked = x * mask_f

        weights = x_masked[:, :, None]
        z_weighted = z_components * weights
        z_sum = jnp.sum(z_weighted, axis=1)

        x_total = jnp.sum(x_masked, axis=1, keepdims=True)
        z_pooled = z_sum / jnp.maximum(x_total, 1e-10)

        h = nn.Dense(128)(z_pooled)
        h = nn.relu(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        return nn.Dense(self.output_dim)(h)


class GE_PredictionHead(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, z_mixture, T, x, mask, training=True):
        T_norm = T / self.cfg.T_REF
        T_feats = jnp.stack([
            T_norm,
            1.0 / jnp.maximum(T_norm, 0.1),
            jnp.log(jnp.maximum(T_norm, 0.1)),
        ], axis=-1)

        features = jnp.concatenate([z_mixture, T_feats], axis=-1)

        h = nn.Dense(self.cfg.GE_HEAD_HIDDEN)(features)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=not training)(h)
        h = nn.Dense(256)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=not training)(h)
        h = nn.Dense(64)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)

        gE_raw = nn.Dense(1)(h).squeeze(-1)

        mask_f = mask.astype(jnp.float32)
        x_masked = x * mask_f
        max_n = x.shape[-1]

        x_i = x_masked[:, :, None]
        x_j = x_masked[:, None, :]
        pairwise = x_i * x_j

        triu = jnp.triu(jnp.ones((max_n, max_n)), k=1)[None, ...]
        pairwise_sum = jnp.sum(pairwise * triu, axis=(1, 2))

        gE_RT = gE_raw * pairwise_sum
        return gE_RT


class HardConstrainedCINN(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, sigma_profiles, scalar_features, mask, T, x, training=True):
        batch_size, max_n = sigma_profiles.shape[:2]

        sigmas_flat = sigma_profiles.reshape(-1, self.cfg.SIGMA_DIM, self.cfg.SIGMA_CHANNELS)
        scalars_flat = scalar_features.reshape(-1, self.cfg.SCALAR_DIM)
        z_flat = ComponentEncoder(cfg=self.cfg)(sigmas_flat, scalars_flat, training=training)
        z_components = z_flat.reshape(batch_size, max_n, self.cfg.LATENT_Z)

        z_mixture = MoleFractionWeightedAggregator(cfg=self.cfg, output_dim=128)(
            z_components, x, mask
        )

        gE_RT = GE_PredictionHead(cfg=self.cfg)(z_mixture, T, x, mask, training=training)
        return gE_RT
