from __future__ import annotations

import jax
import jax.numpy as jnp
import pystl


def _get_semantics(semantics: str, dgmsr_p: int, smooth_temperature: float):
    semantics_kwargs: dict[str, object] = {}
    if semantics == "dgmsr":
        semantics_kwargs["p"] = int(dgmsr_p)
    elif semantics == "smooth":
        semantics_kwargs["temperature"] = float(smooth_temperature)
    elif semantics in {"classical", "agm"}:
        pass
    else:
        raise ValueError(
            f"Unsupported semantics {semantics!r}. "
            "Expected one of: 'dgmsr', 'smooth', 'classical', 'agm'."
        )
    semantics_impl = pystl.create_semantics(semantics, backend="jax", **semantics_kwargs)
    return semantics_impl


def phi_xor_fast(
    traj: jax.Array,
    *,
    eps1: float = 0.1,
    eps2: float = 0.05,
    t1: int = 5,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
) -> jax.Array:
    """
    STL specification implementing an XOR classification
    at steady state.

    Error should be less than ep1 at t1, 
    and eventually less than eps2.

    Expects traj shape (T, 3) with columns [x1, x2, y].
    """
    x_diff = traj[:, 1] - traj[:, 0]
    y_true = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)
    y_pred = traj[:, 2]
    err = jnp.abs(y_true - y_pred)

    interval = (0, int(t1)) if t1 is not None else (0, None)

    err_low = pystl.Predicate("err<eps1", fn=lambda sig, t: eps1 - sig[t])
    err_lowest = pystl.Predicate("err<eps2", fn=lambda sig, t: eps2 - sig[t])

    phi1 = pystl.Eventually(pystl.Always(err_low), interval=interval)
    phi2 = pystl.Eventually(pystl.Always(err_lowest))
    phi = phi1 & phi2

    semantics_impl = _get_semantics(semantics, dgmsr_p, smooth_temperature)
    ro = phi.evaluate(err, semantics_impl, t=0)
    return jnp.asarray(ro).squeeze()



def phi_xor_ss(
    traj: jax.Array,
    *,
    eps: float = 0.1,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
) -> jax.Array:
    """
    STL specification implementing an XOR classification
    at steady state.

    Error should eventually be less than eps.

    Expects traj shape (T, 3) with columns [x1, x2, y].
    """
    x_diff = traj[:, 1] - traj[:, 0]
    y_true = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)
    y_pred = traj[:, 2]
    err = jnp.abs(y_true - y_pred)

    err_low = pystl.Predicate("err<eps1", fn=lambda sig, t: eps - sig[t])

    phi = pystl.Eventually(pystl.Always(err_low))

    semantics_impl = _get_semantics(semantics, dgmsr_p, smooth_temperature)
    ro = phi.evaluate(err, semantics_impl, t=0)
    return jnp.asarray(ro).squeeze()
