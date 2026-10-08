import cmath
from fractions import Fraction

import pytest

from groebner import Ideal, Ring

from .helpers import random_system


def test_membership():
    R = Ring("x,y,z", order="lex")
    I = Ideal(R, ["x - z^2", "y - z^3"])
    assert I.contains("x^3 - y^2")
    assert "x*z - y" in I
    assert not I.contains("x - y")
    assert I.contains(0)


def test_membership_matches_explicit_combinations():
    R, F = random_system(3, order="grevlex")
    I = Ideal(R, F)
    x, y, z = R.gens
    f = (x * y - 3) * F[0] + (z ** 2 + Fraction(1, 2)) * F[1] + F[2] * F[2]
    assert I.contains(f)
    assert not I.contains(f + 1) or I.is_unit()


def test_radical_membership():
    R = Ring("x,y")
    I = Ideal(R, ["x^2", "y^3"])
    assert not I.contains("x + y")
    assert I.radical_contains("x + y")  # (x + y)^4 is in I
    assert not I.radical_contains("x + 1")


def test_unit_and_zero_ideal():
    R = Ring("x,y")
    assert Ideal(R, ["x", "x - 1"]).is_unit()
    assert Ideal(R, [0]).is_zero()
    assert Ideal(R, ["x*y"]) == Ideal(R, ["2*x*y", "x^2*y^2"])
    assert Ideal(R, ["x"]) != Ideal(R, ["y"])


def test_inclusion_and_operations():
    R = Ring("x,y")
    I = Ideal(R, ["x^2", "y"])
    J = Ideal(R, ["x", "y"])
    assert I.is_subset(J) and not J.is_subset(I)
    assert (I + J) == J
    assert (I * J) == Ideal(R, ["x^3", "x*y", "y^2", "x^2*y"])


def test_implicitization_twisted_cubic():
    R = Ring("t,x,y,z", order="grevlex")
    E = Ideal(R, ["x - t", "y - t^2", "z - t^3"]).eliminate(["t"])
    assert E.ring.variables == ("x", "y", "z")
    expected = Ideal(E.ring, ["y - x^2", "z - x^3"])
    assert E == expected


def test_implicitization_of_a_circle_parametrisation():
    # x = (1 - t^2) / (1 + t^2), y = 2t / (1 + t^2), cleared denominators
    R = Ring("t,x,y")
    I = Ideal(R, ["(1 + t^2)*x - (1 - t^2)", "(1 + t^2)*y - 2*t"])
    E = I.eliminate(["t"])
    assert E.contains("x^2 + y^2 - 1")
    assert E == Ideal(E.ring, ["x^2 + y^2 - 1"])


def test_intersection_and_quotient():
    R = Ring("x,y")
    I, J = Ideal(R, ["x"]), Ideal(R, ["y"])
    assert I.intersect(J) == Ideal(R, ["x*y"])
    K = Ideal(R, ["x^2", "x*y"])
    assert K.quotient("x") == Ideal(R, ["x", "y"])
    assert K.quotient("y") == Ideal(R, ["x"])
    assert K.saturate("x") == Ideal(R, [1])
    assert K.saturate("y") == Ideal(R, ["x"])


@pytest.mark.parametrize("seed", range(8))
def test_intersection_contains_products_and_lies_in_both(seed):
    R, F = random_system(seed, nvars=2, npolys=2, order="grevlex", terms=2)
    if len(F) < 2:
        return
    I, J = Ideal(R, [F[0]]), Ideal(R, [F[1]])
    K = I.intersect(J)
    assert K.contains(F[0] * F[1])
    assert K.is_subset(I) and K.is_subset(J)


def test_zero_dimensional_and_degree():
    R = Ring("x,y")
    assert Ideal(R, ["x^2 - 1", "y^2 - 4"]).is_zero_dimensional()
    assert Ideal(R, ["x^2 - 1", "y^2 - 4"]).degree() == 4
    assert not Ideal(R, ["x*y - 1"]).is_zero_dimensional()
    assert Ideal(R, ["x^2", "y"]).degree() == 2  # a double point


def check_solutions(R, gens, sols, tol=1e-8):
    for s in sols:
        pt = [s[v] for v in R.variables]
        for g in gens:
            assert abs(complex(R(g).evaluate(*pt))) < tol


def test_solve_exact_rational_solutions():
    R = Ring("x,y")
    gens = ["x^2 + y^2 - 5", "x*y - 2"]
    sols = Ideal(R, gens).solve()
    assert sols == [
        {"x": -2, "y": -1},
        {"x": -1, "y": -2},
        {"x": 1, "y": 2},
        {"x": 2, "y": 1},
    ]
    assert all(isinstance(v, Fraction) for s in sols for v in s.values())


def test_solve_irrational_and_complex():
    R = Ring("x,y")
    gens = ["x^2 + y^2 - 1", "x - y"]
    sols = Ideal(R, gens).solve()
    assert len(sols) == 2
    check_solutions(R, gens, sols)
    gens = ["x^2 + 1", "y - x"]
    sols = Ideal(R, gens).solve()
    assert len(sols) == 2
    check_solutions(R, gens, sols)
    assert Ideal(R, gens).solve(real_only=True) == []


def test_solve_three_variables_lagrange_multipliers():
    # extrema of x^3 + 2xyz - z^2 on the unit sphere (CLO section 2.8)
    R = Ring("l,x,y,z")
    gens = [
        "3x^2 + 2y*z - 2x*l",
        "2x*z - 2y*l",
        "2x*y - 2z - 2z*l",
        "x^2 + y^2 + z^2 - 1",
    ]
    sols = Ideal(R, gens).solve(real_only=True)
    check_solutions(R, gens, sols, tol=1e-7)
    # CLO finds ten real critical points with x in {-1, -2/3, -3/8, 0, 1}
    assert len(sols) == 10
    xs = sorted({round(float(s["x"]), 9) for s in sols})
    assert xs == [-1.0, round(-2 / 3, 9), -0.375, 0.0, 1.0]
    # points whose every coordinate is rational must come back exact
    exact = [s for s in sols if all(isinstance(v, Fraction) for v in s.values())]
    assert {"l": Fraction(-4, 3), "x": Fraction(-2, 3), "y": Fraction(1, 3), "z": Fraction(2, 3)} in exact
    assert len(exact) == 8  # the two x = -3/8 points have irrational y, z


def test_solve_inconsistent_and_infinite():
    R = Ring("x,y")
    assert Ideal(R, ["x", "x - 1"]).solve() == []
    with pytest.raises(ValueError):
        Ideal(R, ["x*y"]).solve()


def test_graph_three_colouring():
    # Colours are 0, 1, 2. Vertex i: x_i (x_i - 1)(x_i - 2) = 0.
    # Edge (i, j): x_i != x_j, written as (x_i - x_j) * w_ij = 1 is too many
    # variables, so use the classic trick: for values in {0,1,2},
    # x_i != x_j  iff  (x_i - x_j)^2 is 1 or 4, i.e. ((x_i-x_j)^2 - 1)((x_i-x_j)^2 - 4) = 0.
    def colouring_ideal(n, edges):
        R = Ring([f"c{i}" for i in range(n)], order="grevlex")
        gens = [f"c{i}*(c{i} - 1)*(c{i} - 2)" for i in range(n)]
        gens += [f"((c{i} - c{j})^2 - 1)*((c{i} - c{j})^2 - 4)" for i, j in edges]
        return Ideal(R, gens)

    triangle = colouring_ideal(3, [(0, 1), (1, 2), (0, 2)])
    k4 = colouring_ideal(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    assert not triangle.is_unit()
    assert triangle.degree() == 6  # 3! proper colourings
    assert k4.is_unit()  # K4 is not 3 colourable
