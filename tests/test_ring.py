from fractions import Fraction

import pytest

from groebner import ParseError, Ring, block_order


def test_orders_match_cox_little_oshea():
    # CLO, section 2.2: x*y^2*z^... examples.
    a = (1, 2, 1)  # x y^2 z
    b = (0, 3, 2)  # y^3 z^2
    lex, grlex, grevlex = (Ring("x,y,z", order=o).key for o in ("lex", "grlex", "grevlex"))
    assert lex(a) > lex(b)          # x beats anything without x
    assert grlex(b) > grlex(a)      # total degree first
    c = (4, 1, 3)  # x^4 y z^3
    d = (4, 2, 2)  # x^4 y^2 z^2  (same total degree 8)
    assert grlex(c) < grlex(d)      # ties broken by lex
    assert grevlex(c) < grevlex(d)  # smaller power of last variable wins
    e = (1, 5, 2)  # x y^5 z^2
    f = (4, 1, 3)  # x^4 y z^3
    assert grlex(f) > grlex(e)
    assert grevlex(e) > grevlex(f)  # this is where grlex and grevlex differ


def test_block_order_is_an_elimination_order():
    key = block_order(1)
    assert key((1, 0, 0)) > key((0, 9, 9))
    assert key((0, 2, 0)) > key((0, 1, 0))


def test_parse_and_print_round_trip():
    R = Ring("x,y,z", order="grlex")
    for s in [
        "x^2*y - 3/2*z + 1",
        "-x^3 + x*y*z - 7",
        "x",
        "0",
        "5/3",
    ]:
        p = R(s)
        assert R(str(p)) == p
    assert str(R("(x + y)^2")) == "x^2 + 2*x*y + y^2"
    assert R("3x^2y") == R("3*x^2*y")
    assert R("2(x+1)") == R("2*x + 2")
    assert R("0.25 x") == R("x/4")
    assert R("x**3") == R("x^3")
    assert R("-x^2") == -R("x^2")


@pytest.mark.parametrize("bad", ["x +", "x ^ y", "x / y", "w + 1", "(x", "x $ y", "x / 0"])
def test_parse_errors(bad):
    R = Ring("x,y")
    with pytest.raises((ParseError, ZeroDivisionError)):
        R(bad)


def test_arithmetic_identities():
    R = Ring("x,y")
    x, y = R.gens
    f = x ** 3 - 2 * x * y + Fraction(1, 3)
    g = x * y - 1
    assert (f + g) - g == f
    assert f * g - g * f == 0
    assert (f * g) / g == f
    assert (x - y) * (x + y) == x ** 2 - y ** 2
    assert f.evaluate(2, 3) == Fraction(8 - 12) + Fraction(1, 3)
    assert f.evaluate(x=2, y=3) == f(2, 3)
    assert f.subs({"x": y}) == y ** 3 - 2 * y * y + Fraction(1, 3)
    assert f.derivative("x") == 3 * x ** 2 - 2 * y
    assert R("6x^2 + 4/3 y").primitive() == R("9x^2 + 2y")


def test_leading_terms_depend_on_order():
    s = "x*y^2 + x^2 + y^3"
    assert str(Ring("x,y", order="lex")(s).LT) == "x^2"
    assert str(Ring("x,y", order="grlex")(s).LT) == "x*y^2"
    assert str(Ring("y,x", order="lex")(s).LT) == "y^3"


def test_ring_conversion():
    R = Ring("x,y")
    S = Ring("y,t,x", order="lex")
    p = R("x^2 y + 3")
    q = p.to_ring(S)
    assert q.to_ring(R) == p
    with pytest.raises(ValueError):
        S("t + x").to_ring(R)


def test_ring_validation():
    with pytest.raises(ValueError):
        Ring("x,x")
    with pytest.raises(ValueError):
        Ring("x", order="banana")
    with pytest.raises(ValueError):
        Ring("1x")
