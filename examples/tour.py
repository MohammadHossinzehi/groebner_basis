"""A guided tour: run with `python examples/tour.py` from the repo root."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groebner import Ideal, Ring, Stats, buchberger  # noqa: E402


def heading(t):
    print()
    print(t)
    print("=" * len(t))


heading("1. Division is not enough")
R = Ring("x,y", order="lex")
f = R("x^2*y + x*y^2 + y^2")
from groebner import divide  # noqa: E402

_, r1 = divide(f, [R("x*y - 1"), R("y^2 - 1")])
_, r2 = divide(f, [R("y^2 - 1"), R("x*y - 1")])
print(f"remainder with divisors in one order:   {r1}")
print(f"remainder with divisors swapped:        {r2}")
G = buchberger([R("x*y - 1"), R("y^2 - 1")])
print(f"Groebner basis: {[str(g) for g in G]}")
print(f"remainder modulo the basis (unique):    {Ideal(R, G).reduce(f)}")

heading("2. Implicitization: the curve traced by (t^2, t^3)")
R = Ring("t,x,y")
E = Ideal(R, ["x - t^2", "y - t^3"]).eliminate(["t"])
print("implicit equation:", ", ".join(str(g.primitive()) for g in E.groebner_basis()))

heading("3. Where do a circle and a hyperbola meet?")
R = Ring("x,y")
for s in Ideal(R, ["x^2 + y^2 - 5", "x*y - 2"]).solve():
    print("   " + "  ".join(f"{k} = {v}" for k, v in s.items()))

heading("4. Is the complete graph K4 three colourable?")


def colouring(n, edges):
    R = Ring([f"c{i}" for i in range(n)])
    gens = [f"c{i}*(c{i} - 1)*(c{i} - 2)" for i in range(n)]
    gens += [f"((c{i} - c{j})^2 - 1)*((c{i} - c{j})^2 - 4)" for i, j in edges]
    return Ideal(R, gens)


k4 = colouring(4, [(a, b) for a in range(4) for b in range(a + 1, 4)])
c5 = colouring(5, [(i, (i + 1) % 5) for i in range(5)])
print("K4:", "no (the ideal is <1>)" if k4.is_unit() else "yes")
print("C5:", "no" if c5.is_unit() else f"yes, {c5.degree()} colourings")

heading("5. What the criteria buy you (katsura-4)")
from groebner.benchmarks import katsura  # noqa: E402

_, R, F = katsura(4)
for method in ("naive", "gm"):
    st = Stats()
    from groebner import groebner  # noqa: E402

    groebner(F, method=method, stats=st)
    print("  " + st.summary())
