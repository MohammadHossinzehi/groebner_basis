# groebner_basis

Exact Gröbner bases over the rationals, written from scratch in pure Python with zero dependencies.

Gröbner bases are to systems of polynomial equations what Gaussian elimination is to linear ones. Once you have one, a long list of questions that look hard become mechanical: is this polynomial a consequence of those equations, do two systems describe the same set, does the system have any solution at all, what implicit equation does a parametric curve satisfy, and what are all the solutions. This project implements the whole pipeline, from parsing `"x^2*y - 3/2*z"` to returning exact roots, and it is tested against an independent implementation (SymPy) rather than against itself.

```
$ python -m groebner solve "x^2 + y^2 - 5" "x*y - 2"
# 4 solution(s)
x = -2  y = -1
x = -1  y = -2
x = 1  y = 2
x = 2  y = 1
```

## What is in the box

| Module | What it does |
| --- | --- |
| `ring.py` | `Ring` (variables plus a monomial order: lex, grlex, grevlex, or a block elimination order) and an immutable sparse `Poly` with `Fraction` coefficients, arithmetic, substitution, evaluation, derivatives and change of ring |
| `parser.py` | Recursive descent parser: `3x^2y`, `2(x+1)`, `x**3`, exact decimals (`0.1` is 1/10), clear error positions |
| `division.py` | The multivariate division algorithm (returns quotients so `f = Σ qᵢgᵢ + r` can be checked), normal forms, S polynomials |
| `buchberger.py` | Two Buchberger implementations: a textbook one kept as an oracle, and a production one using the Gebauer and Möller criteria with normal or sugar pair selection; reduced bases; Buchberger's criterion as a checker |
| `ideal.py` | `Ideal`: membership, equality, inclusion, sum, product, radical membership (Rabinowitsch trick), elimination, intersection, quotient `I : f`, saturation, zero dimensionality, standard monomials and the number of solutions |
| `solve.py` | Solving zero dimensional systems by back substitution through a lex basis; exact rational roots where they exist, complex floats otherwise |
| `benchmarks.py` | cyclic n, katsura n and friends, plus a harness that prints how much work the criteria save |

## Quick start

Requires Python 3.8 or newer. Nothing to install for the library itself.

```bash
git clone https://github.com/MohammadHossinzehi/groebner_basis
cd groebner_basis

python -m groebner basis "x^3 - 2*x*y" "x^2*y - 2*y^2 + x" --order grlex --stats
python -m groebner member "x^2" "y^3" --poly "x + y"          # in the radical, not the ideal
python -m groebner eliminate "x - t^2" "y - t^3" --drop t       # prints x^3 - y^2
python -m groebner solve "x^2 + y^2 - 1" "x - y" --real
python -m groebner bench
python examples/tour.py
```

Run the tests:

```bash
pip install pytest sympy     # sympy is only used to cross check results
python -m pytest
```

As a library:

```python
from groebner import Ring, Ideal

R = Ring("x,y,z", order="lex")
I = Ideal(R, ["x - z^2", "y - z^3"])
print([str(g) for g in I.groebner_basis()])
# ['x - z^2', 'y - z^3']
print(I.contains("x^3 - y^2"))      # True
print(I.eliminate(["z"]).groebner_basis()[0])   # x^3 - y^2
```

## A tour of what it can answer

**Does a graph have a proper 3 colouring?** Give every vertex a variable that must be 0, 1 or 2 and every edge a polynomial that vanishes exactly when its endpoints differ. The graph is colourable if and only if the ideal is not the whole ring, and for a colourable graph the number of standard monomials counts the colourings. `examples/tour.py` shows K4 is not 3 colourable (the basis collapses to `1`) and that the 5 cycle has exactly 30 colourings, matching the chromatic polynomial `(k−1)ⁿ + (−1)ⁿ(k−1)`.

**Implicitization.** The parametric curve `(t², t³)` is the zero set of `x³ − y²`. The unit circle parametrised by `((1−t²)/(1+t²), 2t/(1+t²))` comes back as `x² + y² − 1`. Both are one call to `eliminate`.

**Lagrange multipliers.** The extrema of `x³ + 2xyz − z²` on the unit sphere (the running example in Cox, Little and O'Shea section 2.8) come out as ten real critical points; the eight that are rational are returned as exact `Fraction`s.

## How it works and why it is built this way

**Exact arithmetic everywhere it matters.** Coefficients are Python `Fraction`s, so there are no tolerances anywhere in basis computation, membership or elimination. A floating point Gröbner basis is mostly meaningless, because deciding whether a coefficient is zero is the entire problem. Floats appear only at the very end, when the solver has to find roots of univariate polynomials that have no rational roots.

**Two Buchberger algorithms on purpose.** `buchberger_naive` reduces every S polynomial of every pair. It is slow but it is the definition, so it is a trustworthy oracle. `buchberger` uses the Gebauer and Möller installation of Buchberger's two criteria:

* the *product criterion*: if the leading monomials of f and g share no variable, S(f, g) reduces to zero;
* the *chain criterion*: if LM(h) divides lcm(LM f, LM g) and the pairs with h are accounted for, the pair (f, g) is redundant.

Basis elements whose leading monomial becomes divisible by a newer one are dropped as they appear, and pairs are processed by the sugar strategy (smallest "sugar degree" first, ties by the monomial order), which behaves far better than plain normal selection under lex and elimination orders. Both functions finish by computing the reduced basis, which is unique, so the two can be compared with plain `==`.

Here is what the criteria buy on the standard benchmarks (`python -m groebner bench`):

| System | Method | S polys reduced | Reduced to zero | Time |
| --- | --- | ---: | ---: | ---: |
| katsura 4 | naive | 253 | 235 | 1.7 s |
| katsura 4 | Gebauer Möller | 26 | 18 | 0.05 s |
| cyclic 4 | naive | 45 | 39 | 0.01 s |
| cyclic 4 | Gebauer Möller | 8 | 5 | 0.002 s |
| cyclic 5 | Gebauer Möller | 116 | 81 | 0.16 s |
| katsura 5 | Gebauer Möller | 64 | 48 | 0.46 s |

A reduction to zero is pure wasted work, so the ratio in the "reduced to zero" column is the honest measure of the criteria.

**Elimination via a block order, not lex.** `Ideal.eliminate` puts the eliminated variables in a first block and compares each block with grevlex. That is still an elimination order (any monomial containing an eliminated variable beats every monomial that does not), so the elimination theorem applies, but it avoids most of lex's coefficient blowup. Intersection (`t·I + (1−t)·J`, eliminate t), quotient (`(I ∩ ⟨f⟩)/f`) and saturation are built on top of it.

**Solving.** For a zero dimensional ideal a lex basis is triangular. The solver walks from the last variable to the first. While every coordinate found so far is rational it stays exact: it substitutes, takes the exact gcd of the resulting univariate polynomials (their common roots are exactly the valid extensions), makes it square free, and only then hunts for roots. Roots are found with the Aberth method, polished with Newton steps, and any root that has a small denominator rational approximation which satisfies the polynomial *exactly* is returned as a `Fraction`. Once an irrational value enters a branch, the rest of that branch is done in complex floating point with residual checks.

## Testing

406 tests, about six seconds:

* **Textbook ground truth.** Worked examples from Cox, Little and O'Shea, *Ideals, Varieties, and Algorithms*: the division example where swapping divisors changes the remainder, the S polynomial example, the grlex basis of `⟨x³ − 2xy, x²y − 2y² + x⟩`, the twisted cubic, and the Lagrange multiplier system.
* **Oracle agreement.** 120 random systems across all three orders, each computed with Gebauer Möller under both selection strategies and with the naive algorithm; all three reduced bases must be identical and must pass Buchberger's criterion.
* **Independent cross check.** 120 more random systems plus cyclic 4, katsura 3 and lex cyclic 4 are compared with SymPy's `groebner`. Reduced bases are unique, so this is exact equality, not a fuzzy match.
* **Invariants.** Reduced bases are unchanged by shuffling generators or adding redundant ones; every reduced element is monic with no term divisible by another leading monomial; division always satisfies `f = Σ qᵢgᵢ + r` with an irreducible remainder.
* **Known numbers.** The grevlex basis of cyclic 5 has 20 elements and the ideal has 70 solutions with multiplicity; katsura 3 has 8.
* **Algebra.** Intersections lie in both ideals and contain products, quotients and saturations match hand computations, radical membership catches nilpotents, and the solver's answers are substituted back into the original equations.

## Limitations

This is a clear, correct implementation, not a replacement for Singular, Magma or msolve. There is no F4/F5 linear algebra, no modular arithmetic with rational reconstruction, and no FGLM change of order, so lex bases of large systems will be slow due to coefficient growth. The numeric part of the solver is fine for the moderate degrees these exact computations can reach, but it is not a certified root isolator.

## License

MIT
