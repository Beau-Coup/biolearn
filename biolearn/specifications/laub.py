"""
Implementation of ARCH-COMP specification from
"""

from pystl import Always, Eventually, Interval, Not, Predicate

from .common import BaseSpec, get_semantics


class StableConverge(BaseSpec):
    def __init__(
        self,
        semantics: str = "classical",
        dgmsr_p: int = 3,
        smooth_temperature: float = 1.0,
    ):
        self.semantics = get_semantics(semantics, dgmsr_p, smooth_temperature)

        x4_small = Predicate("x4<0.4", fn=lambda sig, t: 0.4 - sig[t, 3]).always(
            Interval(0, 20)
        )

        x4_big = Predicate("x4>3.0", fn=lambda sig, t: sig[t, 3] - 3.0)
        x4_reduce = Predicate("x4<3", fn=lambda sig, t: 3.0 - sig[t, 3]).eventually(
            Interval(0, 4)
        )
        x4_decays = Always(Not(x4_big) | x4_reduce, Interval(0, 16))

        c = 0.9 * 0.6 / 1.4 / 0.8

        x3_like_x1_upper = Predicate(
            "x3=cx1high", fn=lambda sig, t: 0.1 - sig[t, 2] + c * sig[t, 0]
        )
        x3_like_x1_lower = Predicate(
            "x3=cx1low", fn=lambda sig, t: 0.1 + sig[t, 2] - c * sig[t, 0]
        )

        x3_like_x1 = Always(x3_like_x1_lower & x3_like_x1_upper)
        x3_eventually_like_x1 = Eventually(x3_like_x1, Interval(0, 10))

        full = x4_small & x4_decays & x3_eventually_like_x1

        self.spec = full
