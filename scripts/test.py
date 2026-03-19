"""
Minimal Neural ODE in Equinox: learn a 2D spiral vector field.

We generate trajectories from a known ODE (damped rotation), then train
a neural network to recover the vector field from data.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

# Curriculum: (n_steps, epochs) — dt fixed at 0.05, so horizon = n_steps * 0.05
SCHEDULE = [
    (20, 200),  # T=1s, fast compilation, stable grads
    (60, 300),  # T=3s, ~1 time constant
    (150, 300),  # T=7.5s, approaching steady state
    (300, 200),  # T=15s, full horizon, fine-tune
]

# --- Ground-truth system: Hill toggle switch (2 species, mutual inhibition) ---
# dx0/dt = alpha / (1 + x1^n) - delta * x0
# dx1/dt = alpha / (1 + x0^n) - delta * x1
ALPHA, DELTA, HILL_N = 1.0, 0.3, 3.0


def true_vector_field(t, y, args):
    x0, x1 = y
    dx0 = ALPHA / (1.0 + x1**HILL_N) - DELTA * x0
    dx1 = ALPHA / (1.0 + x0**HILL_N) - DELTA * x1
    return jnp.array([dx0, dx1])


def generate_data(key, n_trajs=16, n_steps=300, dt=0.05):
    """Roll out true ODE from random ICs, return (t, trajectories)."""
    ts = jnp.linspace(0, n_steps * dt, n_steps)
    solver = diffrax.Tsit5()
    ics = jr.uniform(key, (n_trajs, 2), minval=0.0, maxval=4.0)

    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    def solve_one(y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(true_vector_field),
            solver,
            t0=0.0,
            t1=ts[-1],
            dt0=dt,
            stepsize_controller=stepsize_controller,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys  # (n_steps, 2)

    trajs = jax.vmap(solve_one)(ics)  # (n_trajs, n_steps, 2)
    return ts, trajs


# --- Learned vector field ---
class NeuralVectorField(eqx.Module):
    layers: list

    def __init__(self, key, hidden=128):
        k1, k2, k3 = jr.split(key, 3)
        self.layers = [
            eqx.nn.Linear(2, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, 2, key=k3),
        ]

    def __call__(self, t, y, args):
        h = y
        h = jax.nn.relu(self.layers[0](h))
        h = jax.nn.relu(self.layers[1](h))
        return self.layers[2](h)


# --- Training ---
def make_loss(model, ts, trajs, dt0=0.05):
    """MSE over predicted vs true trajectories."""
    solver = diffrax.Tsit5()
    term = diffrax.ODETerm(model)

    def predict_one(y0):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    preds = jax.vmap(predict_one)(trajs[:, 0, :])  # integrate from IC
    return jnp.mean((preds - trajs) ** 2)


@eqx.filter_jit
def step(model, opt_state, ts, trajs, optimizer):
    loss, grads = eqx.filter_value_and_grad(make_loss)(model, ts, trajs)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(seed=0, lr=1e-3):
    key = jr.PRNGKey(seed)
    k1, k2 = jr.split(key)

    model = NeuralVectorField(k2)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for n_steps, epochs in tqdm(SCHEDULE, desc="Stages"):
        ts, trajs = generate_data(k1, n_steps=n_steps)
        for epoch in range(epochs):
            model, opt_state, loss = step(model, opt_state, ts, trajs, optimizer)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"  n_steps={n_steps:3d} epoch {epoch:4d} | loss {loss:.6f}")

    # Return full-horizon data for plotting
    ts, trajs = generate_data(k1, n_steps=300)
    return model, ts, trajs


# --- Visualization ---
def plot_results(model, ts, trajs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot a few true vs predicted trajectories
    solver = diffrax.Tsit5()
    term = diffrax.ODETerm(model)

    for i in range(min(4, trajs.shape[0])):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=ts[-1],
            dt0=0.05,
            y0=trajs[i, 0, :],
            saveat=diffrax.SaveAt(ts=ts),
        )
        pred = sol.ys
        axes[0].plot(trajs[i, :, 0], trajs[i, :, 1], "b-", alpha=0.4, lw=1.5)
        axes[0].plot(pred[:, 0], pred[:, 1], "r--", alpha=0.7, lw=1.5)
        axes[0].plot(trajs[i, 0, 0], trajs[i, 0, 1], "ko", ms=4)
    axes[0].set_title("Trajectories: true (blue) vs learned (red)")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # Vector field comparison
    grid = jnp.linspace(0.0, 4.0, 20)
    X, Y = jnp.meshgrid(grid, grid)
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    true_vf = jax.vmap(lambda p: true_vector_field(0.0, p, None))(pts)
    learned_vf = jax.vmap(lambda p: model(0.0, p, None))(pts)

    axes[1].quiver(
        X,
        Y,
        true_vf[:, 0].reshape(X.shape),
        true_vf[:, 1].reshape(X.shape),
        color="blue",
        alpha=0.4,
        label="True",
    )
    axes[1].quiver(
        X,
        Y,
        learned_vf[:, 0].reshape(X.shape),
        learned_vf[:, 1].reshape(X.shape),
        color="red",
        alpha=0.6,
        label="Learned",
    )
    axes[1].legend()
    axes[1].set_title("Vector fields")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("neural_ode_results.png", dpi=150)
    print("Saved neural_ode_results.png")


if __name__ == "__main__":
    model, ts, trajs = train(lr=1e-3)
    plot_results(model, ts, trajs)
