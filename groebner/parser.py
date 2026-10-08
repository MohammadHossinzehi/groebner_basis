"""A small recursive descent parser for polynomial expressions.

Grammar (whitespace is ignored)::

    expr   := term (('+' | '-') term)*
    term   := unary (('*' | '/' | <implicit>) unary)*
    unary  := ('+' | '-') unary | power
    power  := atom (('^' | '**') INT)?
    atom   := NUMBER | NAME | '(' expr ')'

Implicit multiplication is allowed between adjacent factors, so ``3x^2y``
and ``2(x+1)`` both work. Division is only allowed by a non-zero constant.
Decimal literals are read exactly (``0.1`` is 1/10, not a float).
"""

from __future__ import annotations

import re
from fractions import Fraction

from .ring import Poly, Ring

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d*)?|\.\d+)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>\*\*|[-+*/^()]))"
)


class ParseError(ValueError):
    pass


def _tokenize(text: str):
    pos = 0
    out = []
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ParseError(f"unexpected character {text[pos]!r} at position {pos} in {text!r}")
        if m.group("num") is not None:
            out.append(("num", m.group("num"), m.start("num")))
        elif m.group("name") is not None:
            out.append(("name", m.group("name"), m.start("name")))
        else:
            out.append(("op", m.group("op"), m.start("op")))
        pos = m.end()
    out.append(("end", "", len(text)))
    return out


class _Parser:
    def __init__(self, text: str, ring: Ring):
        self.text = text
        self.ring = ring
        self.toks = _tokenize(text)
        self.i = 0
        self.gens = dict(zip(ring.variables, ring.gens))

    def peek(self):
        return self.toks[self.i]

    def take(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, op):
        t = self.take()
        if t[0] != "op" or t[1] != op:
            raise ParseError(f"expected {op!r} at position {t[2]} in {self.text!r}")

    def parse(self) -> Poly:
        p = self.expr()
        t = self.peek()
        if t[0] != "end":
            raise ParseError(f"unexpected {t[1]!r} at position {t[2]} in {self.text!r}")
        return p

    def expr(self) -> Poly:
        p = self.term()
        while True:
            t = self.peek()
            if t[0] == "op" and t[1] in "+-":
                self.take()
                q = self.term()
                p = p + q if t[1] == "+" else p - q
            else:
                return p

    def _starts_factor(self, t) -> bool:
        return t[0] in ("num", "name") or (t[0] == "op" and t[1] == "(")

    def term(self) -> Poly:
        p = self.unary()
        while True:
            t = self.peek()
            if t[0] == "op" and t[1] == "*":
                self.take()
                p = p * self.unary()
            elif t[0] == "op" and t[1] == "/":
                self.take()
                d = self.unary()
                if not d.is_constant() or d.is_zero():
                    raise ParseError(f"can only divide by a non-zero constant in {self.text!r}")
                p = p / d.terms[self.ring.zero_mono]
            elif self._starts_factor(t):
                p = p * self.power()
            else:
                return p

    def unary(self) -> Poly:
        t = self.peek()
        if t[0] == "op" and t[1] in "+-":
            self.take()
            p = self.unary()
            return -p if t[1] == "-" else p
        return self.power()

    def power(self) -> Poly:
        base = self.atom()
        t = self.peek()
        if t[0] == "op" and t[1] in ("^", "**"):
            self.take()
            e = self.take()
            if e[0] != "num" or not e[1].isdigit():
                raise ParseError(f"exponent must be a non-negative integer at position {e[2]}")
            return base ** int(e[1])
        return base

    def atom(self) -> Poly:
        t = self.take()
        if t[0] == "num":
            return self.ring.const(Fraction(t[1]))
        if t[0] == "name":
            if t[1] not in self.gens:
                raise ParseError(
                    f"unknown variable {t[1]!r}; ring variables are {', '.join(self.ring.variables)}"
                )
            return self.gens[t[1]]
        if t[0] == "op" and t[1] == "(":
            p = self.expr()
            self.expect(")")
            return p
        raise ParseError(f"unexpected {t[1] or 'end of input'!r} at position {t[2]} in {self.text!r}")


def parse(text: str, ring: Ring) -> Poly:
    """Parse ``text`` into a polynomial of ``ring``."""
    return _Parser(text, ring).parse()


def guess_variables(texts) -> list:
    """Collect identifiers in order of first appearance across expressions."""
    seen = []
    for t in texts:
        for kind, val, _ in _tokenize(t):
            if kind == "name" and val not in seen:
                seen.append(val)
    return seen
