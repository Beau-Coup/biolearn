import jax
import pystl
from pystl.api import Formula, Semantics, Signal


def get_semantics(semantics: str, dgmsr_p: int, smooth_temperature: float):
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
    semantics_impl = pystl.create_semantics(
        semantics, backend="jax", **semantics_kwargs
    )
    return semantics_impl


class BaseSpec:
    spec: Formula
    semantics: Semantics

    def evaluate(
        self,
        traj: jax.Array,
    ) -> jax.Array:
        rho = self.spec.evaluate(Signal(traj), self.semantics, t=0)
        return rho.squeeze()
