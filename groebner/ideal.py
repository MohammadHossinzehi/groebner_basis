"""Ideals and the questions Groebner bases answer about them.

Everything here reduces to one Groebner basis computation:

* membership:            f in I       iff  NF(f, G) == 0
* equality / inclusion:  compare reduced bases / test generators
* radical membership:    f in rad(I)  iff  1 in I + <1 - t*f>   (Rabinowitsch)
* elimination:           I intersect Q[y] is read off a GB under a block order
* intersection:          I cap J = (t*I + (1 - t)*J) intersect Q[x]
* quotient:              I : f = (1/f) * (I cap <f>)
* dimension zero test:   every variable has a pure power among LM(G)
* solving:               triangular structure of a lex basis + root finding
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .buchberger import Stats, groebner
from .division import normal_form
from .ring import Poly, Ring, block_order


def _fresh(ring: Ring, base: str = "t") -> str:
    name = base
    k = 0
    while name in ring.variables:
        k += 1
        name = f"{base}{k}"
    return name


class Ideal:
    """An ideal of Q[x1..xn] given by generators.

    >>> R = Ring("x,y", order="lex")
    >>> I = Ideal(R, ["x^2 + y^2 - 1", "x - y"])
    >>> [str(g) for g in I.groebner_basis()]
    ['x - y', 'y^2 - 1/2']
    >>> I.contains("2*x*y - 1")
    True
    """

    def __init__(self, ring: Ring, generators: Iterable = (), method: str = "gm"):
        self.ring = ring
        self.generators: List[Poly] = [ring(g) for g in generators]
        self.method = method
        self._gb: Optional[List[Poly]] = None
        self.stats: Optional[Stats] = None

    # basics -----------------------------------------------------------------
    def groebner_basis(self) -> List[Poly]:
        if self._gb is None:
            self.stats = Stats()
            self._gb = groebner(self.generators, method=self.method, stats=self.stats)
        return self._gb

    def __repr__(self):
        return f"Ideal(<{', '.join(map(str, self.generators))}> in {self.ring})"

    def is_unit(self) -> bool:
        """True when the ideal is the whole ring (the system has no complex
        solutions, by the weak Nullstellensatz)."""
        G = self.groebner_basis()
        return len(G) == 1 and G[0].is_constant() and not G[0].is_zero()

    def is_zero(self) -> bool:
        return not self.groebner_basis()

    def reduce(self, f) -> Poly:
        """Canonical representative of f modulo the ideal."""
        return normal_form(self.ring(f), self.groebner_basis())

    def contains(self, f) -> bool:
        return not self.reduce(f)

    __contains__ = contains

    def is_subset(self, other: "Ideal") -> bool:
        return all(other.contains(g) for g in self.generators)

    def __eq__(self, other):
        if not isinstance(other, Ideal):
            return NotImplemented
        if self.ring != other.ring:
            return False
        return self.groebner_basis() == other.groebner_basis()

    def __add__(self, other: "Ideal") -> "Ideal":
        return Ideal(self.ring, self.generators + other.generators, self.method)

    def __mul__(self, other: "Ideal") -> "Ideal":
        return Ideal(
            self.ring, [f * g for f in self.generators for g in other.generators], self.method
        )

    # radical membership -----------------------------------------------------
    def radical_contains(self, f) -> bool:
        """Is some power of f in the ideal? (Rabinowitsch trick)"""
        f = self.ring(f)
        t = _fresh(self.ring)
        S = self.ring.extend([t], order="grevlex")
        gens = [g.to_ring(S) for g in self.generators]
        gens.append(S.one() - S.gen(t) * f.to_ring(S))
        return Ideal(S, gens, self.method).is_unit()

    # elimination --------------------------------------------------------------
    def eliminate(self, variables: Sequence[str]) -> "Ideal":
        """The elimination ideal I intersect Q[remaining variables].

        A block order with the eliminated variables in the first block is
        used, so the basis is computed once and the polynomials free of those
        variables form a Groebner basis of the elimination ideal (the
        elimination theorem).
        """
        variables = list(variables)
        for v in variables:
            if v not in self.ring.variables:
                raise ValueError(f"{v!r} is not a variable of {self.ring}")
        keep = [v for v in self.ring.variables if v not in variables]
        if not keep:
            raise ValueError("cannot eliminate every variable")
        E = Ring(variables + keep, order=block_order(len(variables)))
        G = Ideal(E, [g.to_ring(E) for g in self.generators], self.method).groebner_basis()
        target = Ring(keep, order=self.ring._order_spec)
        survivors = [g for g in G if not (set(g.variables_used()) & set(variables))]
        return Ideal(target, [g.to_ring(target) for g in survivors], self.method)

    # intersection and quotient ----------------------------------------------
    def intersect(self, other: "Ideal") -> "Ideal":
        if self.ring != other.ring:
            raise ValueError("ideals live in different rings")
        t = _fresh(self.ring)
        S = self.ring.extend([t], front=True)
        T = S.gen(t)
        gens = [T * g.to_ring(S) for g in self.generators]
        gens += [(S.one() - T) * g.to_ring(S) for g in other.generators]
        E = Ideal(S, gens, self.method).eliminate([t])
        return Ideal(self.ring, [g.to_ring(self.ring) for g in E.groebner_basis()], self.method)

    def quotient(self, f) -> "Ideal":
        """The colon ideal I : f = { g : g*f in I }."""
        f = self.ring(f)
        if f.is_zero():
            return Ideal(self.ring, [1], self.method)
        J = self.intersect(Ideal(self.ring, [f], self.method))
        return Ideal(self.ring, [g / f for g in J.groebner_basis()], self.method)

    def saturate(self, f, max_steps: int = 50) -> "Ideal":
        """I : f^infinity, computed by repeated quotients until stable."""
        cur = self
        for _ in range(max_steps):
            nxt = cur.quotient(f)
            if nxt == cur:
                return cur
            cur = nxt
        raise RuntimeError("saturation did not stabilise")

    # dimension ---------------------------------------------------------------
    def is_zero_dimensional(self) -> bool:
        """Finitely many complex solutions iff, for each variable x, some
        leading monomial of the basis is a pure power of x."""
        G = self.groebner_basis()
        if self.is_unit():
            return False
        n = self.ring.nvars
        for i in range(n):
            if not any(
                g.LM[i] > 0 and all(e == 0 for j, e in enumerate(g.LM) if j != i) for g in G
            ):
                return False
        return True

    def standard_monomials(self, limit: int = 100000) -> List[tuple]:
        """Monomials not divisible by any LM(G). For a zero dimensional ideal
        there are finitely many; their count is the number of solutions with
        multiplicity, and they form a basis of the quotient ring."""
        if not self.is_zero_dimensional():
            raise ValueError("the quotient ring is infinite dimensional")
        from .ring import mono_divides

        G = self.groebner_basis()
        leads = [g.LM for g in G]
        n = self.ring.nvars
        bounds = []
        for i in range(n):
            bounds.append(
                min(
                    g[i]
                    for g in leads
                    if g[i] > 0 and all(e == 0 for j, e in enumerate(g) if j != i)
                )
            )
        out = []

        def rec(prefix):
            if len(out) > limit:
                raise RuntimeError("too many standard monomials")
            if len(prefix) == n:
                m = tuple(prefix)
                if not any(mono_divides(l, m) for l in leads):
                    out.append(m)
                return
            for e in range(bounds[len(prefix)]):
                rec(prefix + [e])

        rec([])
        out.sort(key=self.ring.key)
        return out

    def degree(self) -> int:
        """Number of complex solutions counted with multiplicity."""
        return len(self.standard_monomials())

    # solving -----------------------------------------------------------------
    def solve(self, tol: float = 1e-9, real_only: bool = False) -> List[Dict[str, object]]:
        """All solutions of a zero dimensional system.

        Uses a lex basis, which the elimination theorem makes triangular:
        the last variable appears alone in some element, the next one
        together with it, and so on. We back substitute, finding roots of
        the univariate polynomials numerically and returning exact Fractions
        whenever an exact check confirms a rational root (this is possible
        as long as every coordinate found so far on that branch is rational).
        """
        from .solve import solve_lex

        lex_ring = self.ring.with_order("lex")
        I = Ideal(lex_ring, [g.to_ring(lex_ring) for g in self.generators], self.method)
        if I.is_unit():
            return []
        if not I.is_zero_dimensional():
            raise ValueError("system has infinitely many solutions (ideal is not zero dimensional)")
        return solve_lex(I.groebner_basis(), tol=tol, real_only=real_only)
