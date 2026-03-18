"""Loss functions for BioLearn experiments."""

from .activations import activations
from .base import BoxDomain, SlackModel, make_integral_loss, make_loss, make_slack_loss
from .slack_relu import slack_relu_ic_loss
from .slack_softmax import slack_softmax_loss
from .soft_relu import make_softrelu_loss


def _make_activation_loss(group_loss):
    def ic_loss(
        specification,
        ts,
        *,
        domain=None,
        n_points=128,
        key=None,
        n_boundary_points=0,
        **kwargs,
    ):
        return make_loss(
            group_loss=group_loss,
            specification=specification,
            ts=ts,
            domain=domain,
            n_points=n_points,
            key=key,
            n_boundary_points=n_boundary_points,
            **kwargs,
        )

    return ic_loss


# Backwards-compatible make_<name>_loss helpers used by train_hk25.py
_g = globals()
for _name, _fn in activations.items():
    _g[f"make_{_name}_loss"] = _make_activation_loss(_fn)
del _g, _name, _fn
