from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from biolearn.specifications.ss_classification import xor_ss_spec


def make_temporal_xor_ss_loss(
    ts: jax.Array,
    *,
    eps1: float = 0.1,
    eps2: float = 0.05,
    t1: int = 5,
):
    @eqx.filter_jit
    def _loss_temporal(syst, xbatch, _ybatch):
        def run_and_evaluate_single(xi):
            y_ss_i, _ = syst.simulate(
                x=xi,
                ts=ts,
                to_ss=False,
                stiff=True,
                max_steps=int(1e6),
                rtol=1e-6,
                atol=1e-6,
                progress_bar=False,
            )
            x_traj = jnp.array([[xi[0], xi[1]]])
            x_traj = jnp.repeat(x_traj, repeats=y_ss_i.shape[0], axis=0)

            y_out = y_ss_i[:, -1][:, None]
            traj = jnp.concatenate([x_traj, y_out], axis=1)

            return xor_ss_spec(traj, eps1=eps1, eps2=eps2, t1=t1)

        ros = jax.vmap(run_and_evaluate_single)(xbatch)
        return jax.nn.relu(-ros).sum()

    return _loss_temporal
