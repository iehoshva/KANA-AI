"""ThermodynamicEngine: g^E, ln_gamma, h^E, s^E, and constraint verification."""

import jax.numpy as jnp
from jax import grad, vmap

from .config import Config


class ThermodynamicEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.R = cfg.R_GAS

    def __hash__(self):
        return hash(self.cfg)

    def __eq__(self, other):
        return isinstance(other, ThermodynamicEngine) and self.cfg == other.cfg

    def predict_gE(self, params, apply_fn, sigmas, scalars, mask, T, x):
        gE_RT = apply_fn({'params': params}, sigmas, scalars, mask, T, x, training=False)
        return gE_RT * self.R * T

    def compute_ln_gamma(self, params, apply_fn, sigmas, scalars, mask, T, n):
        def G_excess_over_RT(n_vec, T_scalar, mask_vec, sigma_vec, scalar_vec):
            n_tot = jnp.sum(n_vec * mask_vec)
            x_vec = (n_vec * mask_vec) / jnp.maximum(n_tot, 1e-10)
            gE_RT = apply_fn(
                {'params': params},
                sigma_vec[None, ...], scalar_vec[None, ...], mask_vec[None, ...],
                jnp.array([T_scalar]), x_vec[None, ...], training=False,
            )[0]
            return n_tot * gE_RT

        batch_grad = vmap(
            lambda ni, Ti, mi, si, ci: grad(G_excess_over_RT)(ni, Ti, mi, si, ci),
            in_axes=(0, 0, 0, 0, 0),
        )
        return batch_grad(n, T, mask, sigmas, scalars)

    def compute_hE(self, params, apply_fn, sigmas, scalars, mask, T, x):
        def gE_over_T_scalar(T_scalar, x_vec, mask_vec, sigma_vec, scalar_vec):
            gE = self.predict_gE(
                params, apply_fn,
                sigma_vec[None, ...], scalar_vec[None, ...],
                mask_vec[None, ...], jnp.array([T_scalar]), x_vec[None, ...],
            )[0]
            return gE / T_scalar

        batch_grad = vmap(
            lambda Ti, xi, mi, si, ci: -Ti**2 * grad(gE_over_T_scalar)(Ti, xi, mi, si, ci),
            in_axes=(0, 0, 0, 0, 0),
        )
        return batch_grad(T, x, mask, sigmas, scalars)

    def compute_sE(self, params, apply_fn, sigmas, scalars, mask, T, x, n):
        gE = self.predict_gE(params, apply_fn, sigmas, scalars, mask, T, x)
        hE = self.compute_hE(params, apply_fn, sigmas, scalars, mask, T, x)
        return (hE - gE) / T

    def gibbs_duhem_residual(self, ln_gamma, x, mask, gE_RT):
        x_masked = x * mask.astype(jnp.float32)
        return jnp.sum(x_masked * ln_gamma, axis=-1) - gE_RT
