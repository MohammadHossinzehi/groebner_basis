"""Command line interface.

    python -m groebner basis   "x^2 + y^2 - 1" "x - y" --order lex
    python -m groebner member  "x^2 - y" "y^2 - 1" --poly "x^4 - 1"
    python -m groebner solve   "x^2 + y^2 - 5" "x*y - 2"
    python -m groebner eliminate "x - t^2" "y - t^3" --drop t
    python -m groebner bench
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction

from .buchberger import Stats, groebner
from .ideal import Ideal
from .parser import ParseError, guess_variables
from .ring import ORDERS, Ring


def _ring(args, extra=()):
    if args.vars:
        names = [v.strip() for v in args.vars.split(",") if v.strip()]
    else:
        names = guess_variables(list(args.polys) + list(extra))
        if not names:
            names = ["x"]
    return Ring(names, order=args.order)


def _fmt(v):
    if isinstance(v, Fraction):
        return str(v)
    if isinstance(v, float):
        return f"{v:.12g}"
    if isinstance(v, complex):
        return f"{v.real:.12g}{v.imag:+.12g}i"
    return str(v)


def cmd_basis(args):
    R = _ring(args)
    st = Stats()
    G = groebner([R(p) for p in args.polys], method=args.method, stats=st)
    print(f"# reduced Groebner basis in {R}")
    for g in G:
        print(g)
    if args.stats:
        print("#", st.summary())


def cmd_member(args):
    R = _ring(args, [args.poly])
    I = Ideal(R, args.polys, method=args.method)
    f = R(args.poly)
    r = I.reduce(f)
    if not r:
        print(f"yes: {f} is in the ideal")
    elif I.radical_contains(f):
        print(f"no, but a power of it is (it lies in the radical); remainder {r}")
    else:
        print(f"no: remainder {r}")


def cmd_solve(args):
    R = _ring(args)
    I = Ideal(R, args.polys, method=args.method)
    sols = I.solve(real_only=args.real)
    if not sols:
        print("no solutions")
        return
    print(f"# {len(sols)} solution(s)")
    for s in sols:
        print("  ".join(f"{k} = {_fmt(v)}" for k, v in s.items()))


def cmd_eliminate(args):
    R = _ring(args)
    drop = [v.strip() for v in args.drop.split(",") if v.strip()]
    E = Ideal(R, args.polys, method=args.method).eliminate(drop)
    print(f"# elimination ideal in {E.ring}")
    for g in E.groebner_basis():
        print(g.primitive())


def cmd_bench(args):
    from .benchmarks import run

    run(include_naive=not args.skip_naive)


def main(argv=None):
    p = argparse.ArgumentParser(prog="groebner", description="Exact Groebner bases over Q.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("polys", nargs="+", help="generators, e.g. 'x^2 + y^2 - 1'")
        sp.add_argument("--vars", help="comma separated variables, largest first (default: order of appearance)")
        sp.add_argument("--order", default="grevlex", choices=sorted(ORDERS))
        sp.add_argument("--method", default="gm", choices=["gm", "naive"])

    sp = sub.add_parser("basis", help="reduced Groebner basis")
    common(sp)
    sp.add_argument("--stats", action="store_true", help="print algorithm counters")
    sp.set_defaults(fn=cmd_basis)

    sp = sub.add_parser("member", help="ideal and radical membership")
    common(sp)
    sp.add_argument("--poly", required=True)
    sp.set_defaults(fn=cmd_member)

    sp = sub.add_parser("solve", help="solve a zero dimensional system")
    common(sp)
    sp.add_argument("--real", action="store_true", help="only real solutions")
    sp.set_defaults(fn=cmd_solve)

    sp = sub.add_parser("eliminate", help="elimination ideal / implicitization")
    common(sp)
    sp.add_argument("--drop", required=True, help="variables to eliminate")
    sp.set_defaults(fn=cmd_eliminate)

    sp = sub.add_parser("bench", help="naive vs Gebauer-Moller on classic systems")
    sp.add_argument("--skip-naive", action="store_true")
    sp.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (ParseError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
