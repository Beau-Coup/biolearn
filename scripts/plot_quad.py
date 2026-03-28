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

train_params = [
    jnp.array(-1.256092, dtype=jnp.float64),
    jnp.array(-0.81904369, dtype=jnp.float64),
    jnp.array(0.21673936, dtype=jnp.float64),
    jnp.array(0.00037989, dtype=jnp.float64),
    jnp.array(8.39692474, dtype=jnp.float64),
    jnp.array(3.86830183, dtype=jnp.float64),
]


key = jr.key(seed)
model = Quadrotor(key)
model = QuadModel(model)

# Replace all learnable leaves with trained values
params, static = eqx.partition(model, eqx.is_array)
params = jax.tree_util.tree_unflatten(
    jax.tree_util.tree_structure(params), train_params
)
model = eqx.combine(params, static)


ts = jnp.arange(0.0, t_final, 0.01)
cfg = SimulateConfig(
    to_ss=False,
    stiff=False,
    throw=True,
    max_steps=int(1e4),
    rtol=1e-6,
    atol=1e-8,
    progress_bar=False,
)

fig, (ax_h, ax_v, ax_f, ax_a) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

spec = HeightMaintain()
colors = plt.cm.tab10.colors[: len(inputs)]
ic_labels = [
    f"$\\Delta h_0={float(x0[4]):.1f},\\ \\Delta \\dot{{h}}_0={float(x0[5]):.1f}$"
    for x0 in inputs
]

bm = jnp.exp(train_params[0])
rm = jnp.exp(train_params[1])
g = 9.81
kp = train_params[4]
kd = train_params[5]
mass = bm + rm
body_inertia = 0.4 * bm * (jnp.exp(2 * train_params[3]))
rotor_inertia = 2.0 * rm * (jnp.exp(2.0 * train_params[2]))

inertia_x = body_inertia + rotor_inertia
inertia_y = body_inertia + rotor_inertia
inertia_y = body_inertia + 2.0 * rotor_inertia

amplitude = 5.0
center = 10.0
for i, x0 in enumerate(inputs):
    y_trace, _ = model.simulate(x0, ts=ts, config=cfg)
    readout = y_trace[:, 4:6]
    controller_f = mass * g - kp * (readout[:, 0] - 1.0) - kd * readout[:, 1]
    controller_f = jnp.tanh((controller_f - center) / amplitude) * amplitude + center

    ax_h.plot(ts, readout[:, 0], color=colors[i], label=ic_labels[i])
    ax_v.plot(ts, readout[:, 1], color=colors[i], label=ic_labels[i])
    ax_f.plot(ts, controller_f, color=colors[i], label=ic_labels[i])
    ddh = controller_f / mass * jnp.cos(y_trace[:, 6]) * jnp.cos(y_trace[:, 8]) - g
    ax_a.plot(ts, ddh, color=colors[i], label=ic_labels[i])
    print(f"{spec.evaluate(readout)}")

# STL spec bounds on height panel
ax_h.axhline(1.4, color="red", linestyle="--", linewidth=1, alpha=0.7, label="STL bound")
ax_h.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax_h.axhspan(0.9, 1.4, alpha=0.07, color="green")
ax_h.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8, label="Target $h=1.0$")

# STL spec bounds on velocity panel
ax_v.axhline(0.1, color="red", linestyle="--", linewidth=1, alpha=0.7, label="STL bound")
ax_v.axhline(-0.1, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax_v.axhspan(-0.1, 0.1, alpha=0.07, color="green")

# Labels
ax_h.set_ylabel("Height $h$ (m)")
ax_h.set_title("Quadrotor Height-Maintain Trajectories")
ax_v.set_ylabel("Vertical velocity $\\dot{h}$ (m/s)")
ax_f.set_ylabel("Thrust $F$ (N)")
ax_a.set_ylabel("Vertical accel $\\ddot{h}$ (m/s$^2$)")
ax_a.set_xlabel("Time (s)")

# Shared legend from the height subplot (has IC lines + bound markers)
handles, labels = ax_h.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 0.98), frameon=True)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("qc_trajectories.png", dpi=150)
plt.show()
print("Saved to qc_trajectories.png")


weight_filter = list(filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(model)))

print(f"Raw model parameters: {[jnp.exp(w) for w in weight_filter]}")
