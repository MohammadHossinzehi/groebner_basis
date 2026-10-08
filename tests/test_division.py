import pytest

from groebner import Ring, divide, normal_form, s_polynomial
from groebner.ring import mono_divides

from .helpers import random_system


def test_cox_little_oshea_example_2_3_2():
    R = Ring("x,y", order="lex")
    f = R("x^2*y + x*y^2 + y^2")
    (q1, q2), r = divide(f, [R("x*y - 1"), R("y^2 - 1")])
    assert q1 == R("x + y")
    assert q2 == R("1")
    assert r == R("x + y + 1")


def test_divisor_order_changes_remainder():
    # same example with divisors swapped gives a different remainder,
    # which is the whole motivation for Groebner bases
    R = Ring("x,y", order="lex")
    f = R("x^2*y + x*y^2 + y^2")
    (q1, q2), r = divide(f, [R("y^2 - 1"), R("x*y - 1")])
    assert r == R("2*x + 1")
    assert q1 * R("y^2 - 1") + q2 * R("x*y - 1") + r == f


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_division_identity_and_remainder_property(seed, order):
    R, F = random_system(seed, order=order, npolys=4, terms=4, max_deg=3)
    f, divisors = F[0] * F[-1] + F[0], F[1:]
    if not divisors:
        return
    qs, r = divide(f, divisors)
    total = r
    for q, g in zip(qs, divisors):
        total = total + q * g
    assert total == f
    for m, _ in r:
        assert not any(mono_divides(g.LM, m) for g in divisors)
    assert normal_form(f, divisors) == r


def test_s_polynomial_cancels_leading_terms():
    R = Ring("x,y", order="grlex")
    f, g = R("x^3*y^2 - x^2*y^3 + x"), R("3*x^4*y + y^2")
    s = s_polynomial(f, g)
    assert s == R("-x^3*y^3 + x^2 - 1/3*y^3")  # CLO section 2.6


def test_divide_by_zero_rejected():
    R = Ring("x")
    with pytest.raises(ZeroDivisionError):
        divide(R("x"), [R.zero()])
