import argparse
import os
import traceback
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import matplotlib.pyplot as plt
import optax
import pandas as pd
from tqdm import tqdm

from biolearn import (
    NFC,
    BioSyst,
    MoormanNFC,
)
from biolearn.losses import *
from biolearn.losses.slack_relu import SlackModel
from biolearn.specifications import xor_ss_spec

SS_CLASS_EXP_RESULTS_PATH = "data/results"

jax.config.update("jax_enable_x64", True)

LOSS_CHOICES = ["xor_ss", "sigmoid", "slack_relu", "silu", "softmax", "logsumexp"]


def _clip_pytree(
    pytree: jt.PyTree,
    clip_filter_fn: Optional[Callable[[jt.PyTree], jt.Union[Any, Sequence[Any]]]],
    min_val: jt.ScalarLike,
    max_val: jt.ScalarLike,
):
    """
    Clipping function for PyTrees.
    Args:
        pytree: pytree to clip.
        min_val: minimum value to clip to.
        max_val: maximum value to clip to.

    Returns:
        pytree: clipped pytree.
    """
    if clip_filter_fn is None:
        new_nfc = jax.tree_util.tree_map(
            lambda x: jnp.clip(x, min_val, max_val),
            pytree,
            is_leaf=eqx.is_inexact_array,
        )
    else:
        new_nfc = eqx.tree_at(
            clip_filter_fn,
            pytree,
            replace_fn=lambda _x: jnp.clip(_x, min_val, max_val),
            is_leaf=eqx.is_inexact_array,
        )

    return new_nfc


def adam(
    lr: jt.ScalarLike,
    transition_steps: jt.ScalarLike = 10,
    decay_rate: jt.ScalarLike = 0.9,
    transition_begin: jt.ScalarLike = 50,
    clip_margin: Tuple[jt.ScalarLike, jt.ScalarLike] = (1e-3, jnp.inf),
    clip_filter: Optional[Callable[[jt.PyTree], jt.Union[Any, Sequence[Any]]]] = None,
):
    """
    Adam optimizer customized to NFCs. It incorporates:
        * non-negative parameters
        * clipping of parameters with custom filter and margins
        * learning rate scheduler.

    Args:
        lr: initial learning rate
        transition_steps: transition steps before the decay rate is applied
        decay_rate: decay rate of the learning rate
        transition_begin: number of steps before the initial decay is applied
        clip_margin: margins to clip the parameters
        clip_filter: function to select a subset of parameters
            on which to apply clipping
    Returns:
        optax.GradientTransformation: Adam optimizer
            with custom learning rate scheduler
    """

    def non_negative():
        def init_fn(params):  # pylint: disable=unused-argument
            return ()

        def update_fn(updates, state, params=None):
            _new_params = eqx.apply_updates(params, updates)
            _non_neg_params = jax.tree_util.tree_map(
                jax.nn.relu, _new_params, is_leaf=eqx.is_inexact_array
            )

            clipped_updates = jax.tree_util.tree_map(
                lambda clipped, old: clipped - old, _non_neg_params, params
            )
            return clipped_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    def clip_params():  # pylint: disable=unused-variable
        def init_fn(params):  # pylint: disable=unused-argument
            return ()

        def update_fn(updates, state, params=None):
            _new_params = eqx.apply_updates(params, updates)
            _non_neg_params = jax.tree_util.tree_map(
                jax.nn.relu, _new_params, is_leaf=eqx.is_inexact_array
            )
            _clipped_params = _clip_pytree(
                _non_neg_params, clip_filter, clip_margin[0], clip_margin[1]
            )

            clipped_updates = jax.tree_util.tree_map(
                lambda clipped, old: clipped - old, _clipped_params, params
            )
            return clipped_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    scheduler = optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        transition_begin=transition_begin,
    )

    optimizer = optax.chain(
        optax.scale_by_adam(),
        # optax.clip_by_global_norm(10.0),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
        non_negative(),
        # clip_params(),
    )

    return optimizer


def compute_grad_mag(grads):
    """Compute the magnitude of gradients."""
    squared_grads = jax.tree_util.tree_map(
        lambda g: jnp.sum(g**2), grads, is_leaf=eqx.is_inexact_array
    )
    total_grad = sum(
        jax.tree_util.tree_leaves(squared_grads, is_leaf=eqx.is_inexact_array)
    )
    grad_mag = jnp.sqrt(total_grad)
    return grad_mag


def compute_difference(pytree1: jt.PyTree, pytree2: jt.PyTree) -> jt.PyTree:
    """Computes the difference (in abs) of two identically structured pytrees"""

    pytree1_params, pytree2_params = (
        eqx.filter(pytree1, eqx.is_array),
        eqx.filter(pytree2, eqx.is_array),
    )

    pytree_sub = jax.tree_util.tree_map(
        lambda x, y: jnp.abs(x - y).sum(),
        pytree1_params,
        pytree2_params,
        is_leaf=eqx.is_inexact_array,
    )
    total_sub = sum(jax.tree_util.tree_leaves(pytree_sub, is_leaf=eqx.is_inexact_array))
    return total_sub


# Toy dataloader
def dataloader(
    arrays: jt.PyTree, batch_size: int, *, key: jt.PRNGKeyArray
) -> Iterator[jt.PyTree]:
    """
    Loads data in batches. This is an infinite generator.
    Args:
        arrays: pytree containing the data.
        batch_size: size of the batches.
        key: random key generator for the data loader
    Returns:

    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)

    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        _, key = jax.random.split(key, 2)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def freeze_parameter_mask(nfc: NFC, parameters: List[str]) -> NFC:
    """
    Creates an NFC-type pytree with a mask that is True only for inexact arrays
    that are not on the parameters list.
    Args:
        nfc: nfc to be masked
        parameters: list of parameters to freeze
    Returns:
        NFC-like pytree with the mask
    """
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, nfc)

    for param in parameters:
        for i, layer in enumerate(mask.layers):
            current_val = getattr(layer.nodes, param)
            current_val_false = jax.tree_util.tree_map(lambda _: False, current_val)
            mask = eqx.tree_at(
                lambda _x, i=i, param=param: getattr(_x.layers[i].nodes, param),
                mask,
                replace=current_val_false,
            )

    return mask


def train(
    model: jt.PyTree,
    optimizer: optax.GradientTransformation,
    epochs: int,
    batch_size: int,
    loss_fn: Callable[[jt.PyTree, jt.Array, jt.Array], jt.Scalar],
    x_train: jt.Array,
    y_train: Optional[jt.Array] = None,
    freeze_mask: Optional[jt.PyTree] = None,
    verbose: bool = True,
    *,
    key: jt.PRNGKeyArray,
) -> Tuple[jt.PyTree, List[float]]:
    """
    Training loop for an NFC model.

    Args:
        model: model to train, normally an NFC, but could be any pytree.
        optimizer: optimizer to use for training.
        epochs: number of training epochs.
        batch_size: batch size.
        loss_fn: loss function to use.
        x_train: training data.
        y_train: target training data.
        freeze_mask: function masking trainable parameters or the model.
            All inexact arrays are trained by default
        verbose: whether to print training info.
        key: random key generator for the data loader

    Returns:
        Tuple of (trained model, loss trajectory).
    """
    if batch_size > x_train.shape[0]:
        raise ValueError("Batch size cannot be larger than the dataset size.")

    freeze_mask = eqx.is_inexact_array if freeze_mask is None else freeze_mask

    def get_loss(_diff_model, _static_model, _x_batch, _y_batch):
        _joint_model = eqx.combine(_diff_model, _static_model)
        _loss = loss_fn(_joint_model, _x_batch, _y_batch)
        return _loss

    @eqx.filter_jit
    def batch_step(_model, _x_batch, _y_batch, _optim_state):
        _diff_model, _static_model = eqx.partition(
            _model, freeze_mask, is_leaf=eqx.is_inexact_array
        )
        _loss, _grads = eqx.filter_value_and_grad(get_loss)(
            _diff_model, _static_model, _x_batch, _y_batch
        )
        _updates, _new_optim_state = optimizer.update(
            _grads,
            _optim_state,
            _diff_model,
            value=_loss,
            grad=_grads,
            value_fn=lambda _x: get_loss(_x, _static_model, _x_batch, _y_batch),
        )
        _new_model = eqx.apply_updates(_model, _updates)
        _delta = compute_difference(_model, _new_model)
        _grad_mag = compute_grad_mag(_grads)
        return _loss, _delta, _grad_mag, _new_model, _new_optim_state

    y_train = y_train if y_train is not None else jnp.zeros(x_train.shape[0])
    data_iter = dataloader((x_train, y_train), batch_size, key=key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_traj = []

    pbar = tqdm(range(epochs), total=epochs, disable=not verbose)
    for _ in pbar:
        x_batch, y_batch = next(data_iter)
        try:
            loss, delta_mag, grad_mag, model, opt_state = batch_step(
                model, x_batch, y_batch, opt_state
            )
        except Exception:  # pylint: disable=broad-except
            tqdm.write(f"Stopped due to error raised: {epochs} epochs")
            traceback.print_exc()
            break

        loss_traj.append(float(loss))
        pbar.set_description(
            f"Loss: {loss:.4f}, Delta Mag: {delta_mag:.4f}, Grad Mag: {grad_mag:.4f}"
        )
    else:
        if verbose:
            tqdm.write(f"Reached the end of the {epochs} epochs")
    return model, loss_traj


class ClassModel(BioSyst):
    """
    Implements a system with an NFC + a fluorescent protein
    """

    nfc: NFC
    k: float = eqx.field(static=True, converter=float)
    delta: float = eqx.field(static=True, converter=float)
    w: float = eqx.field(static=True, converter=float)

    def __init__(self, nfc: NFC, k: float, delta: float, w: float):
        self.nfc = nfc
        self.k = k
        self.delta = delta
        self.w = w

    @eqx.filter_jit
    def diffrax_step(self, t, y, args):
        z, o = y[:-1].reshape(self.nfc.shape), y[-1]
        x_in = args[0].evaluate(t)
        dz = self.nfc.ode_step(x_in, z)
        u = z[-self.nfc.out_nodes :, 0]
        do = self.w * u / (u + self.k) - self.delta * o
        dx = jnp.concat([dz.flatten(), do])
        return dx

    def simulate(
        self,
        x: jax.Array | jax.Array,
        ts: Optional[jax.Array],
        x_ts: Optional[jax.Array] = None,
        to_ss: bool = False,
        stiff: bool = True,
        throw: bool = True,
        max_steps: int = int(1e6),
        rtol: float = 1e-10,
        atol: float = 1e-10,
        max_stepsize: Optional[int] = None,
        progress_bar: bool = True,
    ) -> jnp.ndarray:
        args = (self.nfc._handle_inputs(x, x_ts),)
        y0 = jnp.zeros(self.nfc.n_species + 1)

        ys, sol = self._simulate(
            y0=y0,
            ts=ts,
            to_ss=to_ss,
            args=args,
            stiff=stiff,
            throw=throw,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            max_stepsize=max_stepsize,
            progress_bar=progress_bar,
        )
        return ys, sol


def create_dataset(n_samples: int, max_val=1.0):
    """
    Create a 2D grid dataset for training.
    The dataset is created by generating a grid of points in the range [0, max_val]
    Args:
        n_samples: number of samples to generate (should be a perfect square)
        max_val: maximum value for the grid points
    Returns:
        a tuple containing: the generated dataset (x_train)
            and the grid (grid_x, grid_y)
    """
    n_per_axis = int(jnp.sqrt(n_samples))
    x1 = jnp.linspace(0, max_val, n_per_axis)
    x2 = jnp.linspace(0, max_val, n_per_axis)
    grid_x, grid_y = jnp.meshgrid(x1, x2)
    x_train = jnp.column_stack((grid_x.ravel(), grid_y.ravel()))
    return x_train, (grid_x, grid_y)


def _make_loss_fn(loss_name: str, ts: jax.Array):
    """Create the loss function and optional model wrapper for the given loss type."""
    if loss_name == "xor_ss":
        loss_fn = make_temporal_xor_ss_loss(ts, eps1=0.1, eps2=0.05, t1=5)
        wrap_model = None
    elif loss_name == "sigmoid":
        loss_fn = make_sigmoid_loss(specification=xor_ss_spec, ts=ts)
        wrap_model = None
    elif loss_name == "silu":
        loss_fn = make_silu_loss(specification=xor_ss_spec, ts=ts)
        wrap_model = None
    elif loss_name == "logsumexp":
        loss_fn = make_logsumexp_loss(specification=xor_ss_spec, ts=ts)
        wrap_model = None
    elif loss_name == "silu":
        loss_fn = make_silu_loss(specification=xor_ss_spec, ts=ts)
        wrap_model = None
    elif loss_name == "slack_relu":
        loss_fn = slack_relu_ic_loss(specification=xor_ss_spec, ts=ts)
        wrap_model = SlackModel
    else:
        raise ValueError(
            f"Unknown loss function: {loss_name!r}. Choose from {LOSS_CHOICES}"
        )
    return loss_fn, wrap_model


def train_model(
    biosyst: ClassModel,
    x_train: jax.Array,
    loss_name: str,
    lr: float = 0.05,
    epochs: int = 1200,
    show: bool = False,
    *,
    key: jax.Array,
) -> Tuple[NFC, List[float]]:
    """
    Trains the simulator model given a dataset with parameter sets and
    a list of robustness functions.

    Args:
        biosyst: the NFC to train
        x_train: dataset with the values for x.
        loss_name: which loss function to use (xor_ss, sigmoid, slack_relu).
        key: random key to use for training.
    Returns:

    """
    x_diff = x_train[:, 1] - x_train[:, 0]
    y_train = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)
    grid = jnp.meshgrid(jnp.unique(x_train[:, 0]), jnp.unique(x_train[:, 1]))

    ts = jnp.arange(0, 20, 1.0)

    loss_fn, wrap_model = _make_loss_fn(loss_name, ts)

    trainable = wrap_model(biosyst) if wrap_model is not None else biosyst

    learning_rate_schedule = optax.exponential_decay(
        init_value=lr,  # Starting learning rate
        transition_steps=1000,  # Number of steps before each decay
        decay_rate=0.5,  # Multiplicative factor of decay
        staircase=True,  # If True, learning rate decays in discrete steps
    )

    optimizer = optax.adabelief(learning_rate=learning_rate_schedule)

    try:
        trained_model, loss_traj = train(
            trainable,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=x_train.shape[0],
            x_train=x_train,
            y_train=y_train,
            key=key,
        )
        # Unwrap if we used a model wrapper
        if wrap_model is not None:
            trained_syst = trained_model.model
        else:
            trained_syst = trained_model
    except KeyboardInterrupt:
        trained_syst = biosyst
        loss_traj = [float(jnp.inf)]

    def _count_satisfied(x_batch):
        def _run_single(xi):
            y_ss_i, _ = trained_syst.simulate(
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
            return xor_ss_spec(traj, eps1=0.1, eps2=0.05, t1=5)

        ros = jax.vmap(_run_single)(x_batch)
        satisfied = ros >= 0.0
        return int(satisfied.sum()), int(satisfied.shape[0])

    satisfied_count, total_count = _count_satisfied(x_train)
    print(
        f"Specification satisfied on {satisfied_count}/{total_count} initial conditions."
    )

    if show:
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()

        y_out, _ = jax.vmap(
            lambda _xi: trained_syst.simulate(
                x=_xi,
                ts=None,
                to_ss=True,
                stiff=True,
                max_steps=int(1e6),
                rtol=1e-4,
                atol=1e-4,
                progress_bar=False,
            )
        )(x_train)

        plot(
            x1_grid=grid[0],
            x2_grid=grid[1],
            y=y_train.reshape(grid[0].shape),
            title="Target",
            ax=axes[0],
            colorbar=True,
        )
        plot(
            x1_grid=grid[0],
            x2_grid=grid[1],
            y=y_out.reshape(grid[0].shape),
            title="Predicted",
            ax=axes[1],
            colorbar=True,
        )
        plt.show()

    return trained_syst, loss_traj


def run_training(
    layer_sizes,
    x_train,
    k1,
    k2,
    w,
    epochs: int,
    lr: float,
    loss_name: str,
    show=False,
    *,
    key,
):
    sk1, sk2 = jax.random.split(key, num=2)
    print("Dataset shape: ", x_train.shape)
    nfc = MoormanNFC(
        n_inputs=2,
        layer_sizes=layer_sizes,
        gamma=1000,
        beta=1.0,
        k=k1,
        key=sk1,
    )
    biosyst = ClassModel(nfc=nfc, k=k2, delta=1.0, w=w)

    trained_biosyst, loss_traj = train_model(
        biosyst, x_train, loss_name=loss_name, epochs=epochs, lr=lr, show=show, key=sk2
    )
    return trained_biosyst, loss_traj


def repeat_training(
    layer_sizes,
    x_train,
    n,
    epochs=1200,
    lr=0.05,
    loss_name="xor_ss",
    save_path=None,
    show=False,
    *,
    key,
):
    keys = jax.random.split(key, num=n)

    info = []
    for i, ki in enumerate(keys):
        trained_sim, loss_traj = run_training(
            layer_sizes,
            epochs=epochs,
            lr=lr,
            loss_name=loss_name,
            x_train=x_train,
            key=ki,
            show=show,
            w=5.0,
            k1=0.8,
            k2=0.8,
        )

        info.append([len(loss_traj), loss_traj[-1] if loss_traj else float(jnp.inf)])

    return info


def analyze_layer_sizes(
    layer_sizes_list,
    x_train,
    epochs=1000,
    lr=0.05,
    loss_name="xor_ss",
    n=10,
    save=False,
    show=False,
    *,
    key,
):
    all_info = []
    for layer_sizes in layer_sizes_list:
        arch_str = "-".join(str(l) for l in layer_sizes)
        print(f"******* Training for architecture {arch_str} ********")

        experiment_name = f"runs_arch_{arch_str}"

        model_save_path = os.path.join(SS_CLASS_EXP_RESULTS_PATH, experiment_name)

        _, key = jax.random.split(key, num=2)
        info = repeat_training(
            layer_sizes,
            x_train=x_train,
            n=n,
            key=key,
            epochs=epochs,
            lr=lr,
            loss_name=loss_name,
            save_path=model_save_path,
            show=show,
        )

        info_df = pd.DataFrame(info, columns=["Steps", "final loss"])

        if save:
            df_path = os.path.join(model_save_path, "runs.csv")
            info_df.to_csv(df_path)
        all_info.append(info_df)

    return all_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XOR classification with different loss functions"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=LOSS_CHOICES,
        required=True,
        help="Loss function to use for training",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=121,
        help="Number of samples (should be a perfect square)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    key = jax.random.PRNGKey(args.seed)
    key = jax.random.split(key, num=5)[-1]

    archs = [(2, 1), (2, 1, 1)]

    x_train, _ = create_dataset(n_samples=args.n_samples, max_val=1.0)

    info = analyze_layer_sizes(
        archs,
        x_train=x_train,
        epochs=args.epochs,
        lr=args.lr,
        loss_name=args.loss,
        save=args.save,
        show=args.show,
        n=args.n_seeds,
        key=key,
    )

    import code

    code.interact(local=locals())


if __name__ == "__main__":
    main()
