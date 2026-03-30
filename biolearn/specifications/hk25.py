"""
Implementation of STL specification from Krasowski et al. 2025.
Learning Biomolecular Models using Signal Temporal Logic
"""

from pystl import Always, Interval, Not, Predicate

from .common import BaseSpec, get_semantics


class FastProduce(BaseSpec):
    def __init__(
        self,
        semantics: str = "classical",
        dgmsr_p: int = 3,
        smooth_temperature: float = 1.0,
    ):
        self.semantics = get_semantics(semantics, dgmsr_p, smooth_temperature)

        must_produce_condition = Predicate(
            "x1>0.2", fn=lambda sig, t: sig[t, 0] - 0.2
        ) & Predicate("x2>0.3", fn=lambda sig, t: sig[t, 1] - 0.3)

        produce_result = Predicate(
            "x3>0.5", fn=lambda sig, t: sig[t, 2] - 0.5
        ).eventually(Interval(0, 10)) & Predicate(
            "x4>0.9", fn=lambda sig, t: sig[t, 3] - 0.9
        ).always().eventually(Interval(0, 10))

        must_produce = Always(Not(must_produce_condition) | produce_result)

        inhibit3 = Always(
            Not(Predicate("x4>0.6", fn=lambda sig, t: sig[t, 3] - 0.6))
            | (
                Predicate("x3<0.3", fn=lambda sig, t: 0.3 - sig[t, 2])
                .always()
                .eventually(Interval(0, 20))
            )
        )

        max1 = Predicate("x1<1.5", fn=lambda sig, t: 1.5 - sig[t, 0]).always()
        max2 = Predicate("x2<1.5", fn=lambda sig, t: 1.5 - sig[t, 1]).always()
        max3 = Predicate("x3<1.5", fn=lambda sig, t: 1.5 - sig[t, 2]).always()
        max4 = Predicate("x4<1.5", fn=lambda sig, t: 1.5 - sig[t, 3]).always()

        full = (
            must_produce_condition.eventually()
            & must_produce
            & inhibit3
            & max1
            & max2
            & max3
            & max4
        )

        self.spec = full
