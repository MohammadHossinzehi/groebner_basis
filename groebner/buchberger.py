"""Buchberger's algorithm, two ways.

``buchberger_naive`` is the algorithm exactly as it is first taught: keep
every pair, reduce every S-polynomial, add every non-zero remainder. It is
slow but obviously correct, so the test suite uses it as an oracle.

``buchberger`` is the one you would actually use. It keeps a pair list that
is pruned by the Gebauer-Moller installation of Buchberger's two criteria:

* the product criterion: if LM(f) and LM(g) are coprime, S(f, g) reduces to
  zero and the pair can be skipped;
* the chain criterion: if LM(h) divides lcm(LM f, LM g) and the pairs (f, h)
  and (g, h) are already accounted for, the pair (f, g) is redundant.

Pairs are processed with the normal selection strategy (smallest lcm under
the monomial order first), optionally with the "sugar" tie breaker, and
basis elements whose leading monomial becomes divisible by a newer one are
dropped as they appear. The final answer is always made reduced, so both
functions return the same unique reduced Groebner basis for the same input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .division import normal_form, s_polynomial
from .ring import Poly, mono_coprime, mono_divides, mono_lcm


@dataclass
class Stats:
    """Counters that make the effect of the criteria visible."""

    pairs_created: int = 0
    pairs_reduced: int = 0
    zero_reductions: int = 0
    product_criterion: int = 0
    chain_criterion: int = 0
    basis_peak: int = 0
    seconds: float = 0.0
    algorithm: str = ""

    def summary(self) -> str:
        return (
            f"{self.algorithm}: {self.pairs_reduced} S-polys reduced "
            f"({self.zero_reductions} to zero), {self.pairs_created} pairs created, "
            f"{self.product_criterion} skipped by product criterion, "
            f"{self.chain_criterion} by chain criterion, peak basis {self.basis_peak}, "
            f"{self.seconds:.3f}s"
        )


# ---------------------------------------------------------------------------
# reduced bases
# ---------------------------------------------------------------------------

def reduce_basis(G: Sequence[Poly]) -> List[Poly]:
    """Turn any Groebner basis into the unique reduced one.

    1. drop zero polynomials and make everything monic;
    2. drop g whenever LM(g) is divisible by the LM of another element
       (ties between equal leading monomials keep the first one);
    3. replace every remaining g by its normal form modulo the others.
    The result is sorted by leading monomial, largest first.
    """
    G = [g.monic() for g in G if g]
    if not G:
        return []
    ring = G[0].ring
    G.sort(key=lambda g: ring.key(g.LM))  # small leading monomials first
    minimal: List[Poly] = []
    for i, g in enumerate(G):
        lm = g.LM
        redundant = False
        for j, h in enumerate(G):
            if i == j:
                continue
            if mono_divides(h.LM, lm) and (h.LM != lm or j < i):
                redundant = True
                break
        if not redundant:
            minimal.append(g)
    reduced = []
    for i, g in enumerate(minimal):
        others = minimal[:i] + minimal[i + 1:]
        reduced.append(normal_form(g, others).monic())
    reduced.sort(key=lambda g: ring.key(g.LM), reverse=True)
    return reduced


def is_groebner(G: Sequence[Poly]) -> bool:
    """Buchberger's criterion: G is a Groebner basis iff every S-polynomial
    reduces to zero modulo G."""
    G = [g for g in G if g]
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            if normal_form(s_polynomial(G[i], G[j]), G, full=False):
                return False
    return True


# ---------------------------------------------------------------------------
# naive algorithm (oracle)
# ---------------------------------------------------------------------------

def buchberger_naive(F: Sequence[Poly], stats: Optional[Stats] = None) -> List[Poly]:
    st = stats if stats is not None else Stats()
    st.algorithm = "naive"
    t0 = time.perf_counter()
    G = [f for f in F if f]
    if not G:
        st.seconds = time.perf_counter() - t0
        return []
    pairs = [(i, j) for i in range(len(G)) for j in range(i)]
    st.pairs_created = len(pairs)
    while pairs:
        i, j = pairs.pop(0)
        st.pairs_reduced += 1
        r = normal_form(s_polynomial(G[i], G[j]), G)
        if r:
            G.append(r)
            k = len(G) - 1
            new = [(k, m) for m in range(k)]
            st.pairs_created += len(new)
            pairs.extend(new)
            st.basis_peak = max(st.basis_peak, len(G))
        else:
            st.zero_reductions += 1
    out = reduce_basis(G)
    st.seconds = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Gebauer-Moller algorithm
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Pair:
    sort_key: tuple
    i: int = field(compare=False)
    j: int = field(compare=False)
    lcm: tuple = field(compare=False)


def buchberger(
    F: Sequence[Poly],
    stats: Optional[Stats] = None,
    strategy: str = "sugar",
    reduced: bool = True,
) -> List[Poly]:
    """Groebner basis of the ideal generated by F.

    strategy: "normal" picks the pair with the smallest lcm under the ring
    order; "sugar" (the default) picks the smallest sugar degree first and
    uses the normal strategy as a tie breaker, which behaves much better
    under lex and elimination orders.
    """
    if strategy not in ("normal", "sugar"):
        raise ValueError("strategy must be 'normal' or 'sugar'")
    st = stats if stats is not None else Stats()
    st.algorithm = f"gebauer-moller/{strategy}"
    t0 = time.perf_counter()

    F = [f.monic() for f in F if f]
    if not F:
        st.seconds = time.perf_counter() - t0
        return []
    ring = F[0].ring
    key = ring.key

    polys: List[Poly] = []      # every polynomial ever added, by index
    sugar: List[int] = []       # sugar degree of each
    G: List[int] = []           # indices currently in the basis
    B: List[_Pair] = []         # pending critical pairs

    def make_pair(i: int, j: int) -> _Pair:
        L = mono_lcm(polys[i].LM, polys[j].LM)
        if strategy == "sugar":
            s = max(
                sugar[i] + sum(L) - sum(polys[i].LM),
                sugar[j] + sum(L) - sum(polys[j].LM),
            )
            sk = (s, key(L))
        else:
            sk = (0, key(L))
        return _Pair(sk, i, j, L)

    def update(h: int) -> None:
        nonlocal G, B
        lh = polys[h].LM
        C = [g for g in G]
        D: List[int] = []
        # step 1: chain criterion among the new pairs (h, g)
        while C:
            g1 = C.pop(0)
            l1 = mono_lcm(lh, polys[g1].LM)
            keep = mono_coprime(lh, polys[g1].LM)
            if not keep:
                keep = True
                for g2 in C + D:
                    if mono_divides(mono_lcm(lh, polys[g2].LM), l1):
                        keep = False
                        break
            if keep:
                D.append(g1)
            else:
                st.chain_criterion += 1
        # step 2: product criterion
        E = []
        for g in D:
            if mono_coprime(lh, polys[g].LM):
                st.product_criterion += 1
            else:
                E.append(g)
        # step 3: chain criterion on old pairs
        newB = []
        for p in B:
            li, lj = polys[p.i].LM, polys[p.j].LM
            if (
                mono_divides(lh, p.lcm)
                and mono_lcm(li, lh) != p.lcm
                and mono_lcm(lj, lh) != p.lcm
            ):
                st.chain_criterion += 1
                continue
            newB.append(p)
        for g in E:
            newB.append(make_pair(h, g))
            st.pairs_created += 1
        B = newB
        # step 4: drop basis elements made redundant by h
        G = [g for g in G if not mono_divides(lh, polys[g].LM)]
        G.append(h)
        st.basis_peak = max(st.basis_peak, len(G))

    # insert generators in increasing order of leading monomial (fewer
    # immediate redundancies)
    for f in sorted(F, key=lambda p: key(p.LM)):
        f = normal_form(f, [polys[g] for g in G])
        if not f:
            continue
        polys.append(f.monic())
        sugar.append(f.total_degree())
        update(len(polys) - 1)

    while B:
        B.sort()
        p = B.pop(0)
        st.pairs_reduced += 1
        s = s_polynomial(polys[p.i], polys[p.j])
        r = normal_form(s, [polys[g] for g in G])
        if r:
            polys.append(r.monic())
            sugar.append(p.sort_key[0] if strategy == "sugar" else r.total_degree())
            update(len(polys) - 1)
        else:
            st.zero_reductions += 1

    basis = [polys[g] for g in G]
    out = reduce_basis(basis) if reduced else basis
    st.seconds = time.perf_counter() - t0
    return out


def groebner(F: Sequence[Poly], method: str = "gm", stats: Optional[Stats] = None, **kw) -> List[Poly]:
    """Front door: method is "gm" (Gebauer-Moller, default) or "naive"."""
    if method == "naive":
        return buchberger_naive(F, stats=stats)
    if method == "gm":
        return buchberger(F, stats=stats, **kw)
    raise ValueError("method must be 'gm' or 'naive'")
