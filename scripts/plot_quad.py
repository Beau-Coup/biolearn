"""Plot MoormanNFC trajectories for different initial conditions."""

import jax

from biolearn.models.quadrotor import QuadModel, Quadrotor
from biolearn.specifications.quadrotor import HeightMaintain

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from biolearn.models.base import SimulateConfig

# ── Parameters (edit here) ──────────────────────────────────────────
seed = 42
t_final = 5.0
dt = 0.1

inputs = [
    jnp.zeros(12),
    jnp.zeros(12).at[jnp.array([4, 5])].set(jnp.array([0.4, 0.4])),
    jnp.zeros(12).at[jnp.array([4, 5])].set(jnp.array([0.4, -0.4])),
    jnp.zeros(12).at[jnp.array([4, 5])].set(jnp.array([-0.4, 0.4])),
    jnp.zeros(12).at[jnp.array([4, 5])].set(jnp.array([-0.4, -0.4])),
]

key = jr.key(seed)
model = Quadrotor(key)
model = QuadModel(model)

ts = jnp.arange(0.0, t_final, 1.0)
cfg = SimulateConfig(
    to_ss=False,
    stiff=False,
    throw=True,
    max_steps=int(1e4),
    rtol=1e-6,
    atol=1e-8,
    progress_bar=False,
)

fig, ax = plt.subplots(figsize=(8, 5))

spec = HeightMaintain()

for x0 in inputs:
    y_trace, _ = model.simulate(x0, ts=ts, config=cfg)
    # Last node, first species = readout
    readout = y_trace[:, 4]
    ax.plot(ts, readout, label=f"({x0[4]:.1f})")
    print(f"{spec.evaluate(readout)}")

ax.set_xlabel("Time")
ax.set_ylabel("Readout (height)")
ax.set_title("Quadcopter trajectories")
ax.legend(title="")
fig.tight_layout()
fig.savefig("qc_trajectories.png", dpi=150)
plt.show()
print("Saved to qc_trajectories.png")

weight_filter = list(filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(model)))

print(f"Raw model parameters: {[jnp.exp(w) for w in weight_filter]}")
