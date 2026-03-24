"""Write a t raining script from scratch to easily tune things to have a story for the paper and then bring it all back."""

import math
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from biolearn.models.quadrotor import QuadModel, Quadrotor
from biolearn.specifications.quadrotor import HeightMaintain

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from biolearn.models.base import BioModel, SimulateConfig
from biolearn.utils import sample_hypercube_faces


class SamplingBuffer:
    buffer: jax.Array

    def __init__(self, dimension: int, capacity: int = 512):
        self.buffer = jnp.empty((capacity, dimension))
        self.size = 0
        self.capacity = capacity
        self.add_to = 0

    def insert(self, x: jax.Array):
        """Add samples
        x: (B, n)
        """
        if x.ndim == 1:
            x = x.reshape((1, -1))

        len = x.shape[0]
        if self.add_to + len > self.capacity:
            limit = self.capacity - self.add_to

            self.buffer = self.buffer.at[self.add_to : self.capacity].set(x[:limit])

            len = len - limit
            self.add_to = 0
            x = x[limit:]

        self.buffer = self.buffer.at[self.add_to : self.add_to + len].set(x)
        self.add_to = (self.add_to + len) % self.capacity
        self.size = max(self.add_to, self.capacity)

    def sample(self, key: jax.Array, n_samples: int) -> jax.Array:
        indices = jr.choice(key, self.size, (n_samples,))
        return self.buffer[indices]


class MLP(eqx.Module):
    layers: list

    def __init__(self, key: jax.Array, state_size: int, hidden_size: int = 32):
        k1, k2, k3 = jr.split(key, 3)
        out_layer = eqx.nn.Linear(hidden_size, state_size, key=k3)
        self.layers = [
            eqx.nn.Linear(state_size, hidden_size, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, hidden_size, key=k2),
            jax.nn.relu,
            out_layer,
        ]

    def __call__(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class BufferModel(eqx.Module):
    """A module wrapper that adds a residual NN to the ODE."""

    nominal_model: BioModel
    residual_model: MLP
    slack: jax.Array

    def __init__(self, key: jax.Array, nominal: BioModel):
        self.nominal_model = nominal
        state_size = math.prod(nominal.shape)
        residual_model = MLP(key, state_size=state_size, hidden_size=64)
        self.residual_model = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else x,
            residual_model,
        )
        self.slack = jnp.array(0.01)

    def _step(self, t, y, args):
        mlp_out = self.residual_model(y.flatten()).reshape(y.shape)
        return self.nominal_model.diffrax_step(t, y, args) + mlp_out

    def simulate(
        self,
        x: jax.Array,
        ts: jax.Array,
        x_ts=None,
        config: SimulateConfig = SimulateConfig(),
    ):
        y0 = x

        solver = diffrax.Kvaerno5() if config.stiff else diffrax.Tsit5()

        stepsize_controller = diffrax.PIDController(
            pcoeff=0.3 if config.stiff else 0.0,
            icoeff=0.3 if config.stiff else 1.0,
            dcoeff=0,
            rtol=config.rtol,
            atol=config.atol,
            dtmax=config.max_stepsize,
            dtmin=1e-10,
        )
        term = diffrax.ODETerm(self._step)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=config.max_steps,
            throw=config.throw,
        )
        return solution.ys, solution


def residual_l2(model: BufferModel) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(model.residual_model, eqx.is_inexact_array)
    )
    return sum((jnp.sum(w**2) for w in leaves), jnp.zeros(()))


def run_one(key: jax.Array, lr: float, n_epochs: int, reg_weight: float = 1e-4):
    key, subkey = jr.split(key)
    quadrotor = Quadrotor(subkey)
    nominal_model = QuadModel(quadrotor)
    key, subkey = jr.split(key)

    model = BufferModel(subkey, nominal_model)

    spec = HeightMaintain()

    sim_cfg = SimulateConfig(
        to_ss=False,
        stiff=False,
        throw=True,
        max_steps=int(3e4),
        rtol=1e-3,
        atol=1e-4,
        max_stepsize=0.5,
        progress_bar=False,
    )

    ts = jnp.arange(0.0, 5.0, 1.0)

    fast_schedule = optax.exponential_decay(lr, 500, 0.6, staircase=True)
    slow_schedule = optax.exponential_decay(lr * 0.05, 500, 0.6, staircase=True)

    params, static = eqx.partition(model, eqx.is_array)
    labels = jax.tree_util.tree_map(lambda _: "standard", params)
    labels = eqx.tree_at(lambda p: p.slack, labels, "slow")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {
                "standard": optax.adabelief(fast_schedule),
                "slow": optax.adabelief(slow_schedule),
            },
            labels,
        ),
    )

    def ss_to_traj(y_trace):
        y_traj = y_trace[..., 4]  # (B, T, 1)
        return y_traj

    relu_loss = lambda rhos: jax.nn.relu(1e-6 - rhos).mean()
    soft_relu_loss = lambda rhos: (
        jax.nn.relu(-rhos) + 1e-2 * jax.nn.sigmoid(jax.nn.relu(rhos))
    ).mean()
    slack_relu_loss = lambda rhos, epsilon: (
        jax.nn.relu(epsilon - rhos) - 0.5 * epsilon
    ).mean()
    leaky_relu_loss = lambda rhos: jax.nn.leaky_relu(-rhos).mean()

    @eqx.filter_value_and_grad(has_aux=True)
    def grad_loss(model, x0, ts, reg=reg_weight):
        sim = partial(model.simulate, ts=ts, config=sim_cfg)
        y_traces, _ = jax.vmap(sim)(x0)
        safe_traj = ss_to_traj(y_traces)
        rhos = jax.vmap(spec.evaluate)(safe_traj)
        task_loss = relu_loss(rhos)
        return task_loss + reg * residual_l2(model), rhos

    importance_buffer = SamplingBuffer(12, capacity=1024)

    # TODO: Change the dataset
    xs = jnp.linspace(-0.4, 0.4, 10)
    xs = jnp.meshgrid(*[xs, xs, xs, xs, xs, xs])
    x_test = jnp.stack([x.flatten() for x in xs], axis=-1)
    x_test = jnp.concatenate(
        [x_test, jnp.zeros_like(x_test)], axis=-1
    )  # Add zero for angular ICs

    t_horizons = [5.0]

    assert n_epochs % len(t_horizons) == 0, (
        f"Number of epochs ({n_epochs}) can't be split into len(t_horizons)={len(t_horizons)} chunks evenly."
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    all_losses = []
    n_points = 256
    n_boundary = n_points // 4
    n_inside = n_points - n_boundary
    reg_gamma = reg_weight ** (1 / n_epochs)

    for t_final in tqdm(t_horizons, desc="Horizons"):
        ts = jnp.arange(0.0, t_final, 1.0)

        pbar = tqdm(range(n_epochs // len(t_horizons)), desc=f"t={t_final}")
        for i in pbar:
            # Sample points
            key, subkey = jr.split(key)
            edge_samples = sample_hypercube_faces(
                subkey, jnp.ones(6) * -0.4, 0.4 * jnp.ones(6), n_per_face=1
            )

            key, subkey = jr.split(key)
            xs = jnp.concatenate(
                [
                    edge_samples,
                    jr.uniform(subkey, (n_inside, 6), minval=-0.4, maxval=0.4),
                ]
            )
            xs = jnp.concatenate([xs, jnp.zeros_like(xs)], axis=1)

            # Append importance samples
            # if importance_buffer.size > 32:
            #     key, subkey = jr.split(key)
            #     samples = importance_buffer.sample(subkey, 32)
            #     xs = jnp.concatenate([xs, samples], axis=0)

            (loss, rhos), grads = grad_loss(
                model, xs, ts, reg=0.05 * (reg_gamma ** (n_epochs - i))
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            tqdm.set_description(pbar, f"{loss:.2e}")
            model = eqx.apply_updates(model, updates)

            # failed = jnp.nonzero(rhos < 0)
            # if failed[0].shape[0] > 0:
            #     importance_buffer.insert(jax.lax.stop_gradient(xs[failed]))
            all_losses.append(float(loss))

    plt.plot(np.arange(len(all_losses)), all_losses)
    plt.yscale("log")
    plt.show()
    kd = jr.key_data(key)
    plt.savefig(f"losses-{kd[0]}{kd[1]}.png")

    sim = partial(model.nominal_model.simulate, ts=ts, config=sim_cfg)
    y_traces, _ = jax.vmap(sim)(x_test)
    traj = ss_to_traj(y_traces)
    rhos = jax.vmap(spec.evaluate)(traj)

    sats = jnp.sum(rhos > 0).astype(int)
    print(rhos.shape, sats.shape)

    print(f"Satisfied {sats}/{x_test.shape[0]} of initial conditions")

    weight_filter = list(
        filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(model.nominal_model))
    )

    print(f"Raw model parameters: {[w for w in weight_filter]}")


def main():
    n_epochs = 500
    lr = 0.01
    n_runs = 3
    reg = 1e-5

    key = jr.key(42)

    for k in jr.split(key, n_runs):
        run_one(k, lr, n_epochs, reg_weight=reg)


if __name__ == "__main__":
    main()
