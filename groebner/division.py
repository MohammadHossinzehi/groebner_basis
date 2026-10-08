"""Multivariate polynomial division and normal forms.

``divide`` is the textbook algorithm from Cox, Little and O'Shea (Ideals,
Varieties, and Algorithms, section 2.3): it returns quotients q_i and a
remainder r with f = sum(q_i * g_i) + r, where no term of r is divisible by
any LM(g_i). The quotients are returned so the identity can be checked, which
the test suite does on every call it makes.

``normal_form`` computes only the remainder and works on a mutable dict, which
is what Buchberger's algorithm needs in its inner loop.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

from .ring import Monomial, Poly, mono_div, mono_divides, mono_mul


def divide(f: Poly, divisors: Sequence[Poly]) -> Tuple[List[Poly], Poly]:
    ring = f.ring
    for g in divisors:
        if g.ring != ring:
            raise ValueError("all polynomials must live in the same ring")
        if g.is_zero():
            raise ZeroDivisionError("cannot divide by the zero polynomial")

    key = ring.key
    lead = [(g.LM, g.LC, g) for g in divisors]
    quotients: List[Dict[Monomial, Fraction]] = [{} for _ in divisors]
    remainder: Dict[Monomial, Fraction] = {}
    p: Dict[Monomial, Fraction] = dict(f.terms)

    while p:
        lm = max(p, key=key)
        lc = p[lm]
        for i, (gm, gc, g) in enumerate(lead):
            if mono_divides(gm, lm):
                qm = mono_div(lm, gm)
                qc = lc / gc
                quotients[i][qm] = quotients[i].get(qm, 0) + qc
                for m, c in g.terms.items():
                    mm = mono_mul(m, qm)
                    v = p.get(mm, 0) - qc * c
                    if v:
                        p[mm] = v
                    else:
                        p.pop(mm, None)
                break
        else:
            remainder[lm] = lc
            del p[lm]

    return [Poly(ring, q) for q in quotients], Poly(ring, remainder, _trusted=True)


def normal_form(f: Poly, basis: Sequence[Poly], full: bool = True) -> Poly:
    """Remainder of f on division by basis.

    With ``full=False`` only the leading term is reduced until it is no
    longer divisible (a "top reduction"); this is enough to decide whether
    an S-polynomial reduces to zero and is cheaper.
    """
    ring = f.ring
    key = ring.key
    lead = [(g.LM, g.LC, g) for g in basis if g]
    p: Dict[Monomial, Fraction] = dict(f.terms)
    remainder: Dict[Monomial, Fraction] = {}

    while p:
        lm = max(p, key=key)
        lc = p[lm]
        for gm, gc, g in lead:
            if mono_divides(gm, lm):
                qm = mono_div(lm, gm)
                qc = lc / gc
                for m, c in g.terms.items():
                    mm = mono_mul(m, qm)
                    v = p.get(mm, 0) - qc * c
                    if v:
                        p[mm] = v
                    else:
                        p.pop(mm, None)
                break
        else:
            if not full:
                remainder.update(p)
                break
            remainder[lm] = lc
            del p[lm]

    return Poly(ring, remainder, _trusted=True)


def s_polynomial(f: Poly, g: Poly) -> Poly:
    """S(f, g) = (L / LT(f)) * f - (L / LT(g)) * g with L = lcm(LM f, LM g)."""
    from .ring import mono_lcm

    L = mono_lcm(f.LM, g.LM)
    return f.mul_term(mono_div(L, f.LM), 1 / f.LC) - g.mul_term(mono_div(L, g.LM), 1 / g.LC)
