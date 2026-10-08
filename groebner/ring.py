"""Polynomial rings, monomial orders and exact rational polynomials.

A monomial is a tuple of non-negative exponents, one per ring variable.
A polynomial is an immutable mapping {monomial: Fraction} with no zero
coefficients. Every polynomial belongs to exactly one Ring, and the ring
decides how monomials are compared (the monomial order).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple

Monomial = Tuple[int, ...]


# ---------------------------------------------------------------------------
# monomial arithmetic
# ---------------------------------------------------------------------------

def mono_mul(a: Monomial, b: Monomial) -> Monomial:
    return tuple(x + y for x, y in zip(a, b))


def mono_div(a: Monomial, b: Monomial) -> Monomial:
    """a / b, assuming b divides a."""
    return tuple(x - y for x, y in zip(a, b))


def mono_divides(b: Monomial, a: Monomial) -> bool:
    """True when b divides a."""
    return all(y <= x for x, y in zip(a, b))


def mono_lcm(a: Monomial, b: Monomial) -> Monomial:
    return tuple(x if x >= y else y for x, y in zip(a, b))


def mono_coprime(a: Monomial, b: Monomial) -> bool:
    """True when gcd(a, b) == 1, i.e. the monomials share no variable."""
    return all(x == 0 or y == 0 for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# monomial orders: each returns a sort key, larger key == larger monomial
# ---------------------------------------------------------------------------

def _lex(m: Monomial):
    return m


def _grlex(m: Monomial):
    return (sum(m), m)


def _grevlex(m: Monomial):
    return (sum(m), tuple(-e for e in reversed(m)))


ORDERS: Dict[str, Callable[[Monomial], tuple]] = {
    "lex": _lex,
    "grlex": _grlex,
    "grevlex": _grevlex,
}


def block_order(split: int, inner: str = "grevlex") -> Callable[[Monomial], tuple]:
    """Elimination order: compare the first `split` variables with `inner`,
    break ties with `inner` on the remaining variables. Any monomial that
    contains one of the first `split` variables is larger than every monomial
    that does not, which is exactly what elimination needs."""
    key = ORDERS[inner]

    def _block(m: Monomial):
        return (key(m[:split]), key(m[split:]))

    _block.__name__ = f"block{split}_{inner}"
    return _block


# ---------------------------------------------------------------------------
# rings
# ---------------------------------------------------------------------------

class Ring:
    """Q[x1, ..., xn] together with a monomial order.

    >>> R = Ring("x,y", order="lex")
    >>> x, y = R.gens
    >>> str((x + y) ** 2)
    'x^2 + 2*x*y + y^2'
    """

    def __init__(self, variables, order="grevlex"):
        if isinstance(variables, str):
            variables = [v.strip() for v in variables.replace(" ", ",").split(",") if v.strip()]
        variables = tuple(variables)
        if not variables:
            raise ValueError("a ring needs at least one variable")
        if len(set(variables)) != len(variables):
            raise ValueError(f"duplicate variable names in {variables}")
        for v in variables:
            if not (v[0].isalpha() or v[0] == "_") or not all(c.isalnum() or c == "_" for c in v):
                raise ValueError(f"invalid variable name {v!r}")
        self.variables: Tuple[str, ...] = variables
        self.nvars = len(variables)
        if isinstance(order, str):
            if order not in ORDERS:
                raise ValueError(f"unknown order {order!r}; choose from {sorted(ORDERS)}")
            self.order_name = order
            self.key = ORDERS[order]
        else:
            self.order_name = getattr(order, "__name__", "custom")
            self.key = order
        self._order_spec = order
        self.zero_mono: Monomial = (0,) * self.nvars

    # identity -------------------------------------------------------------
    def __eq__(self, other):
        return (
            isinstance(other, Ring)
            and self.variables == other.variables
            and self.order_name == other.order_name
        )

    def __hash__(self):
        return hash((self.variables, self.order_name))

    def __repr__(self):
        return f"Ring({','.join(self.variables)}, order={self.order_name})"

    # constructors ----------------------------------------------------------
    @property
    def gens(self) -> Tuple["Poly", ...]:
        out = []
        for i in range(self.nvars):
            m = [0] * self.nvars
            m[i] = 1
            out.append(Poly(self, {tuple(m): Fraction(1)}))
        return tuple(out)

    def gen(self, name: str) -> "Poly":
        return self.gens[self.variables.index(name)]

    def zero(self) -> "Poly":
        return Poly(self, {})

    def one(self) -> "Poly":
        return self.const(1)

    def const(self, c) -> "Poly":
        c = Fraction(c)
        return Poly(self, {self.zero_mono: c} if c else {})

    def __call__(self, value) -> "Poly":
        """Coerce a string, number or Poly into this ring."""
        if isinstance(value, Poly):
            if value.ring == self:
                return value
            return value.to_ring(self)
        if isinstance(value, str):
            from .parser import parse
            return parse(value, self)
        return self.const(value)

    def with_order(self, order) -> "Ring":
        return Ring(self.variables, order)

    def extend(self, new_vars: Sequence[str], order=None, front: bool = False) -> "Ring":
        new_vars = tuple(new_vars)
        variables = new_vars + self.variables if front else self.variables + new_vars
        return Ring(variables, self._order_spec if order is None else order)


# ---------------------------------------------------------------------------
# polynomials
# ---------------------------------------------------------------------------

def _clean(terms: Dict[Monomial, Fraction]) -> Dict[Monomial, Fraction]:
    return {m: c for m, c in terms.items() if c != 0}


class Poly:
    """Immutable sparse polynomial with Fraction coefficients."""

    __slots__ = ("ring", "terms", "_sorted", "_hash")

    def __init__(self, ring: Ring, terms: Dict[Monomial, Fraction], _trusted: bool = False):
        self.ring = ring
        self.terms = terms if _trusted else _clean({m: Fraction(c) for m, c in terms.items()})
        self._sorted: Optional[list] = None
        self._hash: Optional[int] = None

    # ordering helpers -----------------------------------------------------
    def sorted_monomials(self) -> list:
        """Monomials from largest to smallest under the ring order."""
        if self._sorted is None:
            self._sorted = sorted(self.terms, key=self.ring.key, reverse=True)
        return self._sorted

    def is_zero(self) -> bool:
        return not self.terms

    def __bool__(self):
        return bool(self.terms)

    @property
    def LM(self) -> Monomial:
        if not self.terms:
            raise ValueError("the zero polynomial has no leading monomial")
        return self.sorted_monomials()[0]

    @property
    def LC(self) -> Fraction:
        return self.terms[self.LM]

    @property
    def LT(self) -> "Poly":
        m = self.LM
        return Poly(self.ring, {m: self.terms[m]}, _trusted=True)

    def total_degree(self) -> int:
        return max((sum(m) for m in self.terms), default=-1)

    def degree(self, var) -> int:
        i = self._var_index(var)
        return max((m[i] for m in self.terms), default=-1)

    def variables_used(self) -> Tuple[str, ...]:
        used = [False] * self.ring.nvars
        for m in self.terms:
            for i, e in enumerate(m):
                if e:
                    used[i] = True
        return tuple(v for v, u in zip(self.ring.variables, used) if u)

    def is_constant(self) -> bool:
        return all(not any(m) for m in self.terms)

    def is_univariate_in(self, var) -> bool:
        i = self._var_index(var)
        return all(all(e == 0 for j, e in enumerate(m) if j != i) for m in self.terms)

    def _var_index(self, var) -> int:
        if isinstance(var, int):
            return var
        if isinstance(var, Poly):
            (name,) = var.variables_used()
            var = name
        return self.ring.variables.index(var)

    # arithmetic -----------------------------------------------------------
    def _coerce(self, other) -> "Poly":
        if isinstance(other, Poly):
            if other.ring != self.ring:
                raise ValueError(f"ring mismatch: {self.ring} vs {other.ring}")
            return other
        if isinstance(other, (int, Fraction)):
            return self.ring.const(other)
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        t = dict(self.terms)
        for m, c in other.terms.items():
            v = t.get(m, 0) + c
            if v:
                t[m] = v
            else:
                t.pop(m, None)
        return Poly(self.ring, t, _trusted=True)

    __radd__ = __add__

    def __neg__(self):
        return Poly(self.ring, {m: -c for m, c in self.terms.items()}, _trusted=True)

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        return self + (-other)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        return other - self

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        t: Dict[Monomial, Fraction] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                m = mono_mul(m1, m2)
                v = t.get(m, 0) + c1 * c2
                if v:
                    t[m] = v
                else:
                    t.pop(m, None)
        return Poly(self.ring, t, _trusted=True)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            if other == 0:
                raise ZeroDivisionError("polynomial divided by zero")
            return self.scale(Fraction(1) / Fraction(other))
        if isinstance(other, Poly):
            q, r = exact_divmod(self, other)
            if r:
                raise ValueError("polynomial division is not exact; use divide() for remainders")
            return q
        return NotImplemented

    def __pow__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("exponent must be a non-negative integer")
        result = self.ring.one()
        base = self
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def scale(self, c) -> "Poly":
        c = Fraction(c)
        if c == 0:
            return self.ring.zero()
        return Poly(self.ring, {m: v * c for m, v in self.terms.items()}, _trusted=True)

    def mul_term(self, mono: Monomial, coeff) -> "Poly":
        coeff = Fraction(coeff)
        if coeff == 0:
            return self.ring.zero()
        return Poly(
            self.ring,
            {mono_mul(m, mono): c * coeff for m, c in self.terms.items()},
            _trusted=True,
        )

    def monic(self) -> "Poly":
        if not self.terms:
            return self
        return self.scale(1 / self.LC)

    def primitive(self) -> "Poly":
        """Scale to integer coefficients with gcd 1 and positive leading coeff."""
        if not self.terms:
            return self
        from math import gcd

        den = 1
        for c in self.terms.values():
            den = den * c.denominator // gcd(den, c.denominator)
        nums = [int(c * den) for c in self.terms.values()]
        g = 0
        for n in nums:
            g = gcd(g, n)
        s = Fraction(den, g)
        if self.LC < 0:
            s = -s
        return self.scale(s)

    # comparisons / hashing ------------------------------------------------
    def __eq__(self, other):
        if isinstance(other, (int, Fraction)):
            other = self.ring.const(other)
        if not isinstance(other, Poly):
            return NotImplemented
        return self.ring == other.ring and self.terms == other.terms

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.ring, frozenset(self.terms.items())))
        return self._hash

    # evaluation / substitution -------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, *values, **named):
        """Evaluate at a full point. Values can be ints, Fractions, floats or
        complex numbers; exact inputs give an exact Fraction."""
        if values and named:
            raise TypeError("pass either positional or named values")
        if named:
            values = tuple(named[v] for v in self.ring.variables)
        if len(values) != self.ring.nvars:
            raise ValueError(f"expected {self.ring.nvars} values, got {len(values)}")
        total = 0
        for m, c in self.terms.items():
            t = c
            for v, e in zip(values, m):
                if e:
                    t = t * v ** e
            total = total + t
        return total

    def subs(self, mapping: Dict[str, object]) -> "Poly":
        """Substitute exact numbers or polynomials (in the same ring)."""
        result = self.ring.zero()
        idx = {v: i for i, v in enumerate(self.ring.variables)}
        for m, c in self.terms.items():
            term = self.ring.const(c)
            rest = list(m)
            for name, val in mapping.items():
                i = idx[name]
                e = rest[i]
                if e:
                    rest[i] = 0
                    val = self.ring(val) if not isinstance(val, Poly) else val
                    term = term * val ** e
            term = term * Poly(self.ring, {tuple(rest): Fraction(1)}, _trusted=True)
            result = result + term
        return result

    def to_ring(self, ring: Ring) -> "Poly":
        """Re-express the polynomial in another ring by variable name.
        Every variable used must exist in the target ring."""
        pos = []
        for v in self.ring.variables:
            pos.append(ring.variables.index(v) if v in ring.variables else None)
        t = {}
        for m, c in self.terms.items():
            nm = [0] * ring.nvars
            for i, e in enumerate(m):
                if e:
                    if pos[i] is None:
                        raise ValueError(
                            f"variable {self.ring.variables[i]!r} does not exist in {ring}"
                        )
                    nm[pos[i]] = e
            t[tuple(nm)] = c
        return Poly(ring, t, _trusted=True)

    def derivative(self, var) -> "Poly":
        i = self._var_index(var)
        t = {}
        for m, c in self.terms.items():
            if m[i]:
                nm = list(m)
                nm[i] -= 1
                t[tuple(nm)] = c * m[i]
        return Poly(self.ring, t, _trusted=True)

    def univariate_coeffs(self, var) -> list:
        """Dense coefficient list [c0, c1, ..., cd] for a univariate poly."""
        i = self._var_index(var)
        if not self.is_univariate_in(i):
            raise ValueError("polynomial is not univariate in that variable")
        d = self.degree(i)
        out = [Fraction(0)] * (d + 1)
        for m, c in self.terms.items():
            out[m[i]] = c
        return out

    # printing --------------------------------------------------------------
    def _mono_str(self, m: Monomial) -> str:
        parts = []
        for v, e in zip(self.ring.variables, m):
            if e == 1:
                parts.append(v)
            elif e > 1:
                parts.append(f"{v}^{e}")
        return "*".join(parts)

    def __str__(self):
        if not self.terms:
            return "0"
        out = []
        for k, m in enumerate(self.sorted_monomials()):
            c = self.terms[m]
            neg = c < 0
            a = -c if neg else c
            ms = self._mono_str(m)
            if not ms:
                body = str(a)
            elif a == 1:
                body = ms
            else:
                body = f"{a}*{ms}"
            if k == 0:
                out.append(("-" if neg else "") + body)
            else:
                out.append((" - " if neg else " + ") + body)
        return "".join(out)

    def __repr__(self):
        return f"Poly({self})"

    def __iter__(self) -> Iterator[Tuple[Monomial, Fraction]]:
        for m in self.sorted_monomials():
            yield m, self.terms[m]

    def __len__(self):
        return len(self.terms)


def exact_divmod(f: Poly, g: Poly):
    """Divide by a single polynomial: f = q*g + r (multivariate division)."""
    from .division import divide

    (q,), r = divide(f, [g])
    return q, r


def as_polys(ring: Ring, items: Iterable) -> list:
    return [ring(p) for p in items]
