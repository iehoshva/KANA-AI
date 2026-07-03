"""LLE Proving: Michelsen TPD, isoactivity solver, binodal curves, K-value loading."""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from scipy.optimize import root

from kana.config import Config, PipelineConfig
from kana.inference import KANAInference


@dataclass
class LLEResult:
    """Result of LLE tie-line computation."""
    converged: bool
    x_I: Optional[np.ndarray] = None  # Phase I composition
    x_II: Optional[np.ndarray] = None  # Phase II composition
    beta: float = 0.5  # Phase split fraction
    tpd_min: float = 0.0  # Tangent plane distance
    is_stable: bool = True  # True if single phase is stable (no split)
    n_iterations: int = 0


class LLESolver:
    """Liquid-Liquid Equilibrium solver for ABS mode validation."""

    def __init__(self, inference: KANAInference, pipe_cfg: PipelineConfig):
        self.inference = inference
        self.pipe_cfg = pipe_cfg
        self.cfg = inference.cfg
        self.R = self.cfg.R_GAS
        self.max_n = self.cfg.MAX_COMPONENTS

    def check_phase_stability(self, z_feed: np.ndarray, T: float,
                               sigma: jnp.ndarray, scalar: jnp.ndarray,
                               mask: jnp.ndarray) -> Tuple[bool, float]:
        """Michelsen Tangent Plane Distance (TPD) test.

        If TPD(x) < 0 for ANY trial x, the feed z is unstable (will phase split).

        Returns:
            (is_stable, tpd_min)
        """
        z_jax = jnp.array(z_feed)

        # Compute ln_gamma at feed composition
        ln_gamma_z = self.inference.predict_ln_gamma(
            sigma[None], scalar[None], mask[None],
            jnp.array([T]), z_jax[None],
        )[0]

        # Try multiple trial compositions
        trials = self._generate_trial_compositions(z_feed, mask)
        tpd_values = []

        for x_trial in trials:
            tpd = self._tangent_plane_distance(
                x_trial, z_jax, T, sigma, scalar, mask, ln_gamma_z
            )
            tpd_values.append(float(tpd))

        tpd_min = min(tpd_values) if tpd_values else 0.0
        is_stable = tpd_min >= -1e-8

        return is_stable, tpd_min

    def _tangent_plane_distance(self, x_trial: jnp.ndarray, z_feed: jnp.ndarray,
                                 T: float, sigma: jnp.ndarray, scalar: jnp.ndarray,
                                 mask: jnp.ndarray, ln_gamma_z: jnp.ndarray) -> float:
        """Compute TPD at trial composition x."""
        ln_gamma_x = self.inference.predict_ln_gamma(
            sigma[None], scalar[None], mask[None],
            jnp.array([T]), x_trial[None],
        )[0]

        active = mask.astype(jnp.float32)
        x_clipped = jnp.clip(x_trial, 1e-12, 1.0)
        z_clipped = jnp.clip(z_feed, 1e-12, 1.0)

        term = x_clipped * (jnp.log(x_clipped) + ln_gamma_x
                            - jnp.log(z_clipped) - ln_gamma_z)
        return jnp.sum(active * term)

    def _generate_trial_compositions(self, z_feed: np.ndarray,
                                      mask: jnp.ndarray) -> List[jnp.ndarray]:
        """Generate trial compositions for TPD test."""
        n_active = int(mask.sum())
        trials = []

        # Trial: near-pure components
        for i in range(n_active):
            x = jnp.zeros(self.max_n)
            x = x.at[i].set(0.99)
            for j in range(n_active):
                if j != i:
                    x = x.at[j].set(0.01 / max(n_active - 1, 1))
            trials.append(x)

        # Trial: equimolar
        x_eq = jnp.zeros(self.max_n)
        for i in range(n_active):
            x_eq = x_eq.at[i].set(1.0 / n_active)
        trials.append(x_eq)

        return trials

    def solve_tie_line(self, z_feed: np.ndarray, T: float,
                        sigma: jnp.ndarray, scalar: jnp.ndarray,
                        mask: jnp.ndarray,
                        n_components: int) -> Optional[LLEResult]:
        """Solve for equilibrium tie-line compositions.

        Isoactivity condition: x_i^I * gamma_i^I = x_i^II * gamma_i^II

        Returns:
            LLEResult with converged compositions or None
        """
        # First check phase stability
        is_stable, tpd_min = self.check_phase_stability(z_feed, T, sigma, scalar, mask)
        if is_stable:
            return LLEResult(converged=False, tpd_min=tpd_min, is_stable=True)

        # Initial guess from Rachford-Rice
        n = n_components
        K_inf = self._estimate_K_infinite(z_feed, T, sigma, scalar, mask, n)
        x_I0, x_II0 = self._rachford_rice_guess(z_feed, K_inf, n)

        x0 = np.concatenate([x_I0[:n], x_II0[:n]])

        try:
            sol = root(
                lambda x: np.array(self._isoactivity_residual(
                    jnp.array(x), z_feed, T, sigma, scalar, mask, n
                )),
                x0,
                method='hybr',
                options={'maxfev': 2000, 'xtol': 1e-10},
            )

            if sol.success:
                x_I = np.array(sol.x[:n])
                x_II = np.array(sol.x[n:2*n])
                # Normalize
                x_I = np.clip(x_I, 0, 1)
                x_I = x_I / x_I.sum()
                x_II = np.clip(x_II, 0, 1)
                x_II = x_II / x_II.sum()
                beta = self._compute_beta(z_feed[:n], x_I, x_II)

                return LLEResult(
                    converged=True,
                    x_I=x_I,
                    x_II=x_II,
                    beta=beta,
                    tpd_min=tpd_min,
                    is_stable=False,
                    n_iterations=sol.nfev,
                )
        except Exception:
            pass

        return LLEResult(converged=False, tpd_min=tpd_min, is_stable=False)

    def _isoactivity_residual(self, x_split: jnp.ndarray, z_feed: np.ndarray,
                                T: float, sigma: jnp.ndarray, scalar: jnp.ndarray,
                                mask: jnp.ndarray, n: int) -> jnp.ndarray:
        """Compute isoactivity residual for Newton-Raphson solver."""
        x_I = x_split[:n]
        x_II = x_split[n:2*n]

        # Pad to MAX_COMPONENTS
        x_I_pad = jnp.zeros(self.max_n)
        x_I_pad = x_I_pad.at[:n].set(x_I)
        x_II_pad = jnp.zeros(self.max_n)
        x_II_pad = x_II_pad.at[:n].set(x_II)

        ln_gamma_I = self.inference.predict_ln_gamma(
            sigma[None], scalar[None], mask[None],
            jnp.array([T]), x_I_pad[None],
        )[0]
        ln_gamma_II = self.inference.predict_ln_gamma(
            sigma[None], scalar[None], mask[None],
            jnp.array([T]), x_II_pad[None],
        )[0]

        # Isoactivity: x_i^I * gamma_i^I = x_i^II * gamma_i^II
        r_iso = (jnp.log(jnp.clip(x_I, 1e-12)) + ln_gamma_I[:n]
                 - jnp.log(jnp.clip(x_II, 1e-12)) - ln_gamma_II[:n])

        # Closure
        r_sum_I = jnp.sum(x_I) - 1.0
        r_sum_II = jnp.sum(x_II) - 1.0

        # Mass balance
        beta = self._compute_beta_jax(z_feed[:n], x_I, x_II)
        r_mass = beta * x_I[0] + (1 - beta) * x_II[0] - z_feed[0]

        return jnp.concatenate([r_iso, jnp.array([r_sum_I, r_sum_II, r_mass])])

    def _estimate_K_infinite(self, z_feed: np.ndarray, T: float,
                              sigma: jnp.ndarray, scalar: jnp.ndarray,
                              mask: jnp.ndarray, n: int) -> np.ndarray:
        """Estimate K values from infinite dilution activity coefficients."""
        K = np.ones(n)
        for i in range(n):
            x_trial = np.zeros(self.max_n)
            x_trial[i] = 1.0
            lg = self.inference.predict_ln_gamma(
                sigma[None], scalar[None], mask[None],
                jnp.array([T]), jnp.array(x_trial[None]),
            )[0]
            K[i] = np.exp(-float(lg[i]))
        return K

    def _rachford_rice_guess(self, z: np.ndarray, K: np.ndarray,
                              n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initial guess from Rachford-Rice flash."""
        beta = 0.5
        for _ in range(20):
            f = np.sum(z[:n] * (K - 1) / (1 + beta * (K - 1)))
            df = -np.sum(z[:n] * (K - 1)**2 / (1 + beta * (K - 1))**2)
            if abs(df) < 1e-30:
                break
            beta = beta - f / df
            beta = np.clip(beta, 0.01, 0.99)

        x_I = z[:n] * K / (1 + beta * (K - 1))
        x_II = z[:n] / (1 + beta * (K - 1))
        x_I = np.clip(x_I, 1e-10, 1.0)
        x_II = np.clip(x_II, 1e-10, 1.0)
        x_I = x_I / x_I.sum()
        x_II = x_II / x_II.sum()

        return x_I, x_II

    @staticmethod
    def _compute_beta(z: np.ndarray, x_I: np.ndarray, x_II: np.ndarray) -> float:
        """Compute phase split fraction from lever rule."""
        denom = x_I - x_II
        mask = np.abs(denom) > 1e-10
        if mask.any():
            betas = z[mask] / denom[mask]
            return float(np.median(betas))
        return 0.5

    @staticmethod
    def _compute_beta_jax(z: jnp.ndarray, x_I: jnp.ndarray, x_II: jnp.ndarray) -> float:
        """JAX version of beta computation."""
        denom = x_I - x_II
        mask = jnp.abs(denom) > 1e-10
        betas = jnp.where(mask, z / denom, 0.5)
        return jnp.median(betas)

    def check_gibbs_mixing_concavity(self, sigma: jnp.ndarray, scalar: jnp.ndarray,
                                       mask: jnp.ndarray, T: float,
                                       x_I: np.ndarray, x_II: np.ndarray) -> bool:
        """Check that ΔG_mix is concave in the miscibility gap.

        Valid LLE: all interior points below the common tangent line.
        """
        n_points = 20
        n = int(mask.sum())
        results = []

        for alpha in np.linspace(0.05, 0.95, n_points):
            x = alpha * x_I[:n] + (1 - alpha) * x_II[:n]
            x_pad = np.zeros(self.max_n)
            x_pad[:n] = x

            gE_RT = self.inference.predict_gE_RT(
                sigma[None], scalar[None], mask[None],
                jnp.array([T]), jnp.array(x_pad[None]),
            )

            x_clipped = np.clip(x, 1e-12, 1.0)
            delta_G = float(gE_RT[0]) + np.sum(x_clipped * np.log(x_clipped))
            results.append(delta_G)

        # Common tangent line
        x_I_clipped = np.clip(x_I[:n], 1e-12, 1.0)
        x_II_clipped = np.clip(x_II[:n], 1e-12, 1.0)

        gE_I = float(self.inference.predict_gE_RT(
            sigma[None], scalar[None], mask[None],
            jnp.array([T]), jnp.array(np.concatenate([x_I, np.zeros(self.max_n - n)])[None]),
        )[0])
        gE_II = float(self.inference.predict_gE_RT(
            sigma[None], scalar[None], mask[None],
            jnp.array([T]), jnp.array(np.concatenate([x_II, np.zeros(self.max_n - n)])[None]),
        )[0])

        dG_I = gE_I + np.sum(x_I_clipped * np.log(x_I_clipped))
        dG_II = gE_II + np.sum(x_II_clipped * np.log(x_II_clipped))

        # Check all interior points are below tangent
        for i, alpha in enumerate(np.linspace(0.05, 0.95, n_points)):
            tangent_val = alpha * dG_I + (1 - alpha) * dG_II
            if results[i] > tangent_val + 1e-4:
                return False

        return True
