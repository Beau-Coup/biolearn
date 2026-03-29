"""Plot BioGNN (Hill model) trajectories for different initial conditions."""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from biolearn.models.base import SimulateConfig
from biolearn.models.hill import BioGNN, BioGnnModel, EdgeType
from biolearn.specifications.hk25 import FastProduce

# ── Graph topology (HK25) ───────────────────────────────────────────
graph = [
    (0, 1, EdgeType.Activation),  # x1 -> x2
    (2, 3, EdgeType.Inhibition),  # x3 -| x4
    (3, 0, EdgeType.Inhibition),  # x4 -| x1
    (3, 1, EdgeType.Activation),  # x4 -> x2
    (4, 1, EdgeType.Activation),  # x5 -> x2
    (4, 2, EdgeType.Activation),  # x5 -> x3
    (4, 5, EdgeType.Activation),  # x5 -> x6
]

key = jr.key(0)
gnn = BioGNN(key, graph, 2.0)
model = BioGnnModel(gnn)

# ── Load trained parameters ─────────────────────────────────────────
trained_params = [
    jnp.array([0.09494472], dtype=jnp.float64),
    jnp.array([], dtype=jnp.float64),
    jnp.array([], dtype=jnp.float64),
    jnp.array([0.21040559, 0.91751554, 0.36848164], dtype=jnp.float64),
    jnp.array([], dtype=jnp.float64),
    jnp.array([0.43391442], dtype=jnp.float64),
    jnp.array([0.39582614], dtype=jnp.float64),
    jnp.array([], dtype=jnp.float64),
    jnp.array([], dtype=jnp.float64),
    jnp.array([0.08597344], dtype=jnp.float64),
    jnp.array(
        [0.82392755, 0.45774357, 0.67599819, 0.39003168, 0.55302649, 0.92344576],
        dtype=jnp.float64,
    ),
    jnp.array(
        [0.38168326, 0.12809297, 0.45018981, 0.02895972, 0.18048485, 0.70098302],
        dtype=jnp.float64,
    ),
    jnp.array(
        [0.41958752, 0.15026935, 0.56776042, 0.09449914, 0.63742486, 0.21226839],
        dtype=jnp.float64,
    ),
]

# Replace all learnable leaves with trained values
params, static = eqx.partition(model, eqx.is_inexact_array)
params = jax.tree_util.tree_unflatten(
    jax.tree_util.tree_structure(params), trained_params
)
model = eqx.combine(params, static)

# ── Simulation config ───────────────────────────────────────────────
ts = jnp.arange(0, 25, 0.1)
cfg = SimulateConfig(
    to_ss=False,
    stiff=True,
    throw=True,
    max_steps=int(1e6),
    rtol=1e-6,
    atol=1e-6,
    progress_bar=False,
)

# ── Initial conditions ──────────────────────────────────────────────
inputs = [
    jnp.array([0.01, 0.01, 0.01, 0.01, 0.99, 0.99]),
    jnp.array([0.02, 0.02, 0.01, 0.01, 0.99, 0.99]),
    jnp.array([0.00, 0.0, 0.0, 0.0, 1.0, 1.0]),
]

# ── Plot ─────────────────────────────────────────────────────────────
spec = FastProduce()
species_names = ["x1", "x2", "x3", "x4", "x5", "x6"]

fig, axes = plt.subplots(len(inputs), 1, figsize=(10, 3 * len(inputs)), sharex=True)

for i, x0 in enumerate(inputs):
    y_trace, _ = model.simulate(x0, ts=ts, config=cfg)
    rho = spec.evaluate(y_trace)
    print(f"IC {i} (x0={x0}): robustness = {rho}")

    ax = axes[i]
    for j, name in enumerate(species_names):
        ax.plot(ts, y_trace[:, j], label=name)
    ax.set_ylabel("Concentration")
    ax.set_title(f"IC {i}: robustness = {float(rho):.4f}")
    ax.legend(loc="upper right", ncol=6, fontsize=8)

axes[-1].set_xlabel("Time")
fig.suptitle("BioGNN (Hill model) trajectories", fontsize=14)
fig.tight_layout()
fig.savefig("hill_trajectories.png", dpi=150)
plt.show()
print("Saved to hill_trajectories.png")

# ── Raw parameters ──────────────────────────────────────────────────
weight_filter = list(filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(model)))
print(f"Raw model parameters: {[jnp.exp(w) for w in weight_filter]}")
