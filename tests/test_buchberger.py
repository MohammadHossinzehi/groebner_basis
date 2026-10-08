import random

import pytest

from groebner import (
    Ideal,
    Ring,
    Stats,
    buchberger,
    buchberger_naive,
    is_groebner,
    normal_form,
    reduce_basis,
)
from groebner.benchmarks import cyclic, katsura

from .helpers import random_system


def strs(G):
    return [str(g) for g in G]


def test_cox_little_oshea_example_2_7_1():
    R = Ring("x,y", order="grlex")
    G = buchberger([R("x^3 - 2*x*y"), R("x^2*y - 2*y^2 + x")])
    assert strs(G) == ["x^2", "x*y", "y^2 - 1/2*x"]


def test_twisted_cubic_lex():
    R = Ring("x,y,z", order="lex")
    G = buchberger([R("y - x^2"), R("z - x^3")])
    assert strs(G) == ["x^2 - y", "x*y - z", "x*z - y^2", "y^3 - z^2"]


def test_inconsistent_system_gives_unit_ideal():
    R = Ring("x,y")
    G = buchberger([R("x*y - 1"), R("x"), R("y + 3")])
    assert strs(G) == ["1"]


def test_empty_and_zero_inputs():
    R = Ring("x,y")
    assert buchberger([]) == []
    assert buchberger([R.zero()]) == []
    assert buchberger_naive([R.zero(), R.zero()]) == []


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_gm_agrees_with_naive_oracle(seed, order):
    R, F = random_system(seed, order=order)
    a = buchberger(F, strategy="sugar")
    b = buchberger(F, strategy="normal")
    c = buchberger_naive(F)
    assert a == b == c
    assert is_groebner(a)
    # every generator lies in the ideal of the basis and vice versa
    for f in F:
        assert not normal_form(f, a)


@pytest.mark.parametrize("seed", range(15))
def test_reduced_basis_is_unique_under_input_permutation(seed):
    R, F = random_system(seed, nvars=3, npolys=4, order="grevlex")
    G = buchberger(F)
    rng = random.Random(seed)
    for _ in range(3):
        H = list(F)
        rng.shuffle(H)
        # throw in redundant generators too
        H.append(H[0] * H[-1] + H[1])
        assert buchberger(H) == G


@pytest.mark.parametrize("seed", range(15))
def test_reduced_basis_properties(seed):
    R, F = random_system(seed, order="lex")
    G = buchberger(F)
    lms = [g.LM for g in G]
    assert len(set(lms)) == len(lms)
    for g in G:
        assert g.LC == 1
        others = [h for h in G if h is not g]
        # no term of g is divisible by another leading monomial
        for m, _ in g:
            assert not any(all(a <= b for a, b in zip(h.LM, m)) for h in others)


@pytest.mark.parametrize("seed", range(10))
def test_reduce_basis_of_padded_unreduced_basis(seed):
    R, F = random_system(seed, order="grlex")
    raw = buchberger(F, reduced=False)
    assert is_groebner(raw)
    # pad with scaled copies and ideal multiples: still a Groebner basis
    padded = [g * 7 for g in raw] + [raw[0] * R.gens[0] + raw[-1]] + raw
    assert reduce_basis(padded) == buchberger(F)


def test_criteria_actually_prune_work():
    name, R, F = cyclic(4)
    s_gm, s_naive = Stats(), Stats()
    a = buchberger(F, stats=s_gm)
    b = buchberger_naive(F, stats=s_naive)
    assert a == b
    assert s_gm.pairs_reduced < s_naive.pairs_reduced / 3
    assert s_gm.chain_criterion > 0 and s_gm.product_criterion > 0


def test_cyclic5_has_twenty_element_grevlex_basis():
    # well known: the reduced grevlex basis of cyclic-5 has 20 elements and
    # the ideal has 70 solutions counted with multiplicity
    name, R, F = cyclic(5)
    I = Ideal(R, F)
    assert len(I.groebner_basis()) == 20
    assert I.is_zero_dimensional()
    assert I.degree() == 70


def test_katsura3_degree():
    # katsura-n has 2^n solutions
    name, R, F = katsura(3)
    assert Ideal(R, F).degree() == 8


def test_bad_arguments():
    R = Ring("x")
    with pytest.raises(ValueError):
        buchberger([R("x")], strategy="random")
