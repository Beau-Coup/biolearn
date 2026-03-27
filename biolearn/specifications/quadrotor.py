"""
Implementation of ARCH-COMP specification from
"""

from pystl import Interval, Predicate

from .common import BaseSpec, get_semantics


class HeightMaintain(BaseSpec):
    def __init__(
        self,
        semantics: str = "classical",
        dgmsr_p: int = 3,
        smooth_temperature: float = 1.0,
    ):
        self.semantics = get_semantics(semantics, dgmsr_p, smooth_temperature)

        max_height = Predicate("h<1.4", fn=lambda sig, t: 1.4 - sig[t, 0]).always()

        min_height = Predicate("h>0.9", fn=lambda sig, t: sig[t, 0] - 0.9).always(
            Interval(1)
        )

        settle_vel = Predicate("hdot>-0.1", fn=lambda sig, t: sig[t, 1] + 0.1).always(
            Interval(3)
        ) & Predicate("hdot<0.1", fn=lambda sig, t: 0.1 - sig[t, 1]).always(Interval(3))

        full = min_height & max_height

        self.spec = full
