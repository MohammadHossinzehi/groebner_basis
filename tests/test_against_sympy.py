"""Cross check against an independent implementation (SymPy's groebner).

SymPy is an optional test dependency; these tests are skipped without it.
Reduced Groebner bases are unique, so the comparison is exact equality of
monic polynomials, not a fuzzy check.
"""

from fractions import Fraction

import pytest

from groebner import Ring, buchberger
from groebner.benchmarks import cyclic, katsura

from .helpers import random_system

sympy = pytest.importorskip("sympy")

_SYMPY_ORDER = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}


def via_sympy(R, F):
    syms = sympy.symbols(R.variables)
    exprs = [sympy.sympify(str(f).replace("^", "**"), locals=dict(zip(R.variables, syms))) for f in F]
    G = sympy.groebner(exprs, *syms, order=_SYMPY_ORDER[R.order_name], domain="QQ")
    out = []
    for g in G.exprs:
        p = sympy.Poly(g, *syms)
        terms = {m: Fraction(int(c.p), int(c.q)) for m, c in p.terms()}
        out.append(R.zero() + type(R.one())(R, terms))
    return sorted((str(g) for g in out))


# Lex bases of random cubics suffer heavy coefficient growth (that is a
# property of lex, not a bug), so cubics are only compared under grevlex.
@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("order,deg", [("lex", 2), ("grlex", 2), ("grevlex", 2), ("grevlex", 3)])
def test_random_systems_match_sympy(seed, order, deg):
    R, F = random_system(seed + 1000, order=order, npolys=3, terms=3, max_deg=deg)
    ours = sorted(str(g) for g in buchberger(F))
    assert ours == via_sympy(R, F)


@pytest.mark.parametrize("make", [lambda: cyclic(4), lambda: katsura(3), lambda: cyclic(4, "lex")])
def test_benchmarks_match_sympy(make):
    name, R, F = make()
    assert sorted(str(g) for g in buchberger(F)) == via_sympy(R, F)
