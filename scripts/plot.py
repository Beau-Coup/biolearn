"""Plot MoormanNFC trajectories for different initial conditions."""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from biolearn.models import MoormanNFC
from biolearn.models.base import SimulateConfig

# ── Parameters (edit here) ──────────────────────────────────────────
n_inputs = 2
layer_sizes = [2, 1]
k = 0.8
gamma = 1000.0
beta = 1.0
seed = 42
t_final = 20.0
dt = 0.1

inputs = [(0.1, 0.3)]

# Learned parameters (exp-space, from training output).
# Set to None to use random initialization instead.
learned_params = [
    jnp.array(
        [
            [[0.8783128, 0.17739959], [0.099926, 0.9955761]],
            [[0.16594264, 1.03253341], [1.13980637, 0.11276796]],
        ]
    ),
    jnp.array([[0.00019125, 0.14035029], [0.00093659, 0.16912786]]),
    jnp.array([[[3.0723347, 2.78159661], [0.0630462, 0.10228087]]]),
    jnp.array([[0.00362009, 0.00290267]]),
]
# ────────────────────────────────────────────────────────────────────

key = jr.key(seed)
model = MoormanNFC(n_inputs, layer_sizes, gamma=gamma, beta=beta, k=k, key=key)

if learned_params is not None:
    # Back to log space
    log_params = [jnp.log(p) for p in learned_params]
    leaves, treedef = jax.tree_util.tree_flatten(model)
    is_leaf = [eqx.is_inexact_array(l) for l in leaves]
    param_idx = 0
    for i, leaf in enumerate(leaves):
        if is_leaf[i]:
            leaves[i] = log_params[param_idx]
            param_idx += 1
    model = jax.tree_util.tree_unflatten(treedef, leaves)

ts = jnp.arange(0.0, t_final, dt)
cfg = SimulateConfig(
    to_ss=False,
    stiff=True,
    throw=True,
    max_steps=int(1e4),
    rtol=1e-6,
    atol=1e-8,
    progress_bar=False,
)

fig, ax = plt.subplots(figsize=(8, 5))

for x1, x2 in inputs:
    x = jnp.array([x1, x2])
    y_trace, _ = model.simulate(x, ts=ts, config=cfg)
    # Last node, first species = readout
    readout = y_trace.reshape((-1, 6))
    ax.plot(ts, readout, label=f"({x1:.1f}, {x2:.1f})")

ax.set_xlabel("Time")
ax.set_ylabel("Readout (last node, species 0)")
ax.set_title("MoormanNFC trajectories")
ax.legend(title="(x1, x2)")
fig.tight_layout()
fig.savefig("trajectories.png", dpi=150)
plt.show()
print("Saved to trajectories.png")

weight_filter = list(filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(model)))

print(f"Raw model parameters: {[jnp.exp(w) for w in weight_filter]}")
