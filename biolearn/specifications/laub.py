"""
Implementation of ARCH-COMP specification from
"""

from pystl import Always, Interval, Not, Predicate

from .common import BaseSpec, get_semantics


class StableConverge(BaseSpec):
    def __init__(
        self,
        semantics: str = "classical",
        dgmsr_p: int = 3,
        smooth_temperature: float = 1.0,
    ):
        self.semantics = get_semantics(semantics, dgmsr_p, smooth_temperature)

        x4_small = Predicate("x4<4.5", fn=lambda sig, t: 4.5 - sig[t, 3]).always(
            Interval(0, 20)
        )

        x4_big = Predicate("x4>3.0", fn=lambda sig, t: sig[t, 3] - 3.0)
        x4_reduce = Predicate("x4<3", fn=lambda sig, t: 3.0 - sig[t, 3]).eventually(
            Interval(0, 4)
        )
        x4_decays = Always(Not(x4_big) | x4_reduce, Interval(0, 16))

        x2_bar = 0.35
        x2_tol = 0.1
        x2_conv_upper = Predicate(
            "x2<x2bar", fn=lambda sig, t: x2_bar + x2_tol - sig[t, 1]
        )
        x2_conv_lower = Predicate(
            "x2>x2bar", fn=lambda sig, t: sig[t, 1] - x2_bar + x2_tol
        )
        x2_conv = (x2_conv_lower & x2_conv_upper).always().eventually(Interval(0, 10))

        x3_bar = 0.55
        x3_tol = 0.1
        x3_conv_upper = Predicate(
            "x3<x3bar", fn=lambda sig, t: x3_bar + x3_tol - sig[t, 2]
        )
        x3_conv_lower = Predicate(
            "x3>x3bar", fn=lambda sig, t: sig[t, 2] - x3_bar + x3_tol
        )
        x3_conv = (x3_conv_lower & x3_conv_upper).always().eventually(Interval(0, 10))

        full = x4_small & x2_conv & x3_conv

        self.spec = full
