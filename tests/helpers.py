import random
from fractions import Fraction

from groebner import Ring


def random_poly(R: Ring, rng: random.Random, terms=3, max_deg=2, coeff=5):
    p = R.zero()
    for _ in range(terms):
        while True:
            m = tuple(rng.randint(0, max_deg) for _ in range(R.nvars))
            if sum(m) <= max_deg:
                break
        c = rng.randint(-coeff, coeff) or 1
        mono = R.one()
        for g, e in zip(R.gens, m):
            mono = mono * g ** e
        p = p + mono * Fraction(c)
    return p


def random_system(seed, nvars=3, npolys=3, order="grevlex", **kw):
    rng = random.Random(seed)
    names = "xyzwuv"[:nvars]
    R = Ring(",".join(names), order=order)
    F = [random_poly(R, rng, **kw) for _ in range(npolys)]
    return R, [f for f in F if f]
