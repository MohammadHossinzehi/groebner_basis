"""Classic benchmark systems and a small comparison harness."""

from __future__ import annotations

from .buchberger import Stats, groebner
from .ring import Ring


def cyclic(n: int, order: str = "grevlex"):
    names = [f"x{i}" for i in range(n)]
    R = Ring(names, order=order)
    x = R.gens
    F = []
    for d in range(1, n):
        s = R.zero()
        for i in range(n):
            t = R.one()
            for k in range(d):
                t = t * x[(i + k) % n]
            s = s + t
        F.append(s)
    prod = R.one()
    for v in x:
        prod = prod * v
    F.append(prod - 1)
    return f"cyclic-{n}", R, F


def katsura(n: int, order: str = "grevlex"):
    names = [f"u{i}" for i in range(n + 1)]
    R = Ring(names, order=order)
    u = R.gens

    def U(i):
        i = abs(i)
        return u[i] if i <= n else R.zero()

    F = []
    for m in range(n):
        s = R.zero()
        for l in range(-n, n + 1):
            s = s + U(l) * U(m - l)
        F.append(s - U(m))
    s = R.zero()
    for l in range(-n, n + 1):
        s = s + U(l)
    F.append(s - 1)
    return f"katsura-{n}", R, F


def twisted_cubic(order: str = "lex"):
    R = Ring("t,x,y,z", order=order)
    F = [R("x - t"), R("y - t^2"), R("z - t^3")]
    return "twisted cubic (lex)", R, F


def systems():
    """(name, ring, generators, run_naive_too)"""
    return [
        twisted_cubic() + (True,),
        cyclic(3) + (True,),
        cyclic(4) + (True,),
        cyclic(4, order="lex") + (True,),
        katsura(3) + (True,),
        katsura(4) + (True,),
        cyclic(5) + (False,),
        katsura(5) + (False,),
    ]


def run(include_naive: bool = True):
    print(f"{'system':<22}{'method':<24}{'S-polys':>8}{'zero':>7}{'chain':>7}{'prod':>6}{'|G|':>5}{'time':>9}")
    for name, R, F, naive_ok in systems():
        if R.order_name != "grevlex" and "lex" not in name:
            name = f"{name} ({R.order_name})"
        methods = ["gm"] + (["naive"] if include_naive and naive_ok else [])
        results = {}
        for m in methods:
            st = Stats()
            G = groebner(F, method=m, stats=st)
            results[m] = G
            print(
                f"{name:<22}{st.algorithm:<24}{st.pairs_reduced:>8}{st.zero_reductions:>7}"
                f"{st.chain_criterion:>7}{st.product_criterion:>6}{len(G):>5}{st.seconds:>8.3f}s"
            )
        if "naive" in results:
            assert results["gm"] == results["naive"], f"bases disagree on {name}"
    if include_naive:
        print("reduced bases from both methods agree on every system where both ran")


if __name__ == "__main__":
    run()
