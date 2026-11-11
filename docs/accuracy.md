# Accuracy

What go-fft promises about the numbers a transform returns, how it is checked,
and the constants. Tracked in [#121](https://github.com/go-fft/fft/issues/121).

## The contract

For every public transform, in float64 and float32, with the default
arithmetic and with `Options.FMA`, the relative RMS error against an
extended-precision oracle is at most

    ‖got − exact‖₂ / ‖exact‖₂  ≤  c · ε · log2(n)

- `n` is the number of points (the product of the shape for N-dimensional
  transforms; `log2 n` is taken as at least 1).
- `ε` is DBL_EPSILON `2^-52` for float64 and FLT_EPSILON `2^-23` for float32.
- `c` is a constant per function family and precision, below.
- "exact" is the oracle's result for the same input: the float64 or float32
  values widened, which is exact.

The contract is an error bound, not equality with another implementation. FFTW
and pocketfft are not bit-identical across machines, plans or builds either,
and the Go reference is itself only identical per architecture (on arm64 gc
fuses multiply-adds). A Cooley–Tukey FFT's error is `O(ε·log n)` whatever the
order of operations (Higham, *Accuracy and Stability of Numerical Algorithms*,
§24.1), so the order and the fusing are free to change inside the bound.

Until the default changes, the arithmetic is also bit-identical to the package's
Go reference on amd64 (README, "Opt-in fused multiply-add"); the tests that
guard that are unchanged.

## The oracle

`oracle_test.go`, test code only.

- A **double-double FFT** (about 106 bits, 1e-32 relative): a mixed-radix
  decimation-in-time recursion in `O(n·Σp)` for `n = Πp`, where a prime factor
  `p` is a direct `O(p²)` DFT. Powers of two and smooth lengths are
  `O(n log n)`; a prime of 1,009 points takes about 0.1 s. The tests go to 4,096
  points for complex lengths and 3,125 for real ones, and to 2,039 for the
  Bluestein and Rader lengths; `-race` and `-short` stop at 1,100.
- Every **twiddle** is computed once per length by `math/big` at 200 bits (a
  Taylor series on `[0, π/4]`, the other angles by exact symmetry) and rounded
  to a double-double, never by a recurrence.
- The **real** (RFFT, IRFFT, HFFT, IHFFT), **N-dimensional** (the 1-D oracle along
  each axis) and **r2r** (DCT/DST I–IV, forward and inverse, backward and ortho
  norm: the direct `O(n²)` sum of the definitions in `dct.go`/`dst.go`, with
  scipy's ortho corrections) oracles are built on the same code.
- The error is formed against the double-double (`(got − hi) − lo`), so the
  oracle's own error never shows.
- It is validated (`TestOracleRoots`, `TestOracleDFT`, `TestOracleR2R`) against
  closed-form roots (`N = 8, 12`), the identities `|w| = 1` and `w[2k] = w_{N/2}[k]`,
  `math.Sincos` to 1e-14, a direct double-double sum for 13 lengths, a pure tone,
  Parseval, the inverse round trip, and a hand-computed DCT-II, all to `1e-29`
  or better.

One trap worth recording: gc on arm64 fuses `x*y + z` across statements, so
`p := a*b` followed by `s := p + e` is not an error-free transformation unless
`p` is written `float64(a*b)`. The oracle does that.

## The constants

Measured by `go test -run TestAccuracyBound -v -accuracy.table`: every public
function of the family (its plan methods and `...With` variants included), at
the lengths and shapes below, with two random seeds (inputs uniform in
`[-1, 1)`), default and FMA. The worst `err/(ε·log2 n)` is in the table; `c` is
the first multiple of 0.25 at or above twice that, so a platform whose rounding
differs by a few tens of percent stays inside.

| family | functions | worst (measured) | c |
|:--|:--|--:|--:|
| `c2c` | FFT, IFFT, Plan.FFT/IFFT, FFTReal | 0.64 | 1.5 |
| `r2c` | RFFT, IRFFT, HFFT, IHFFT, RealPlan | 0.54 | 1.25 |
| `c2cN` | FFTN, IFFTN, FFT2, IFFT2, PlanN | 0.30 | 0.75 |
| `r2cN` | RFFTN, IRFFTN, RFFT2, IRFFT2, RealPlan2 | 0.28 | 0.75 |
| `r2r` | DCT, IDCT, DST, IDST (types I–IV, backward and ortho), their plans, the N-D forms | 0.95 | 2.0 |
| `c2c/32` | FFT32, IFFT32, Plan32, the `...32With` forms | 0.34 | 0.75 |
| `r2c/32` | RFFT32, IRFFT32, HFFT32, IHFFT32, RealPlan32 | 0.43 | 1.0 |
| `c2cN/32` | FFTN32, IFFTN32, FFT2_32, IFFT2_32, PlanN32 | 0.15 | 0.5 |
| `r2cN/32` | RFFTN32, IRFFTN32, RFFT2_32, IRFFT2_32, RealPlan2_32 | 0.14 | 0.5 |
| `r2r/32` | the float32 DCT/DST family | 0.86 | 1.75 |

Measured on arm64 (Apple silicon, Go 1.27); the amd64 kernels and the FMA plans
are asserted by the same test in CI. FMA is not less accurate: its worst ratio
in every family is within 0.05 of the default's (`c2c`: 0.64 against 0.60).
float32 ignores `Options.FMA`. Where the CPU has no FMA3 the test still builds
the FMA plans (its Go reference), so every platform checks them.

Lengths: powers of 2 (2 to 4,096), of 3 (to 2,187), of 5 (to 3,125), of 7 (to
2,401), mixed smooth lengths (6 to 2,310, including 1,000 and 1,001), Rader
primes (17, 31, 127, 257, 1,009), Bluestein lengths (19, 323, 1,013, 2,018,
2,039), odd real lengths (9, 15, 105, 343, 1,001, 1,009, 1,013, 3,125), 2-D to 4-D
shapes (4×4 to 128×128, 30×31, 127×33, 1,009×3, 8×9×10, 16³, 17×3×5×4), and the
r2r lengths 2 to 1,009.

The worst ratios sit at the shortest lengths (19, 3, 2), where `log2 n` is
smallest and the error does not shrink with it. Rader and Bluestein lengths
measure 0.2 to 0.6, the same as the smooth ones.

## What the test sees: the mutation check

`TestAccuracyMutation` perturbs one twiddle of the oracle's own table, at seven
lengths and three positions, and holds the result to the `c2c` bound:

| perturbation | error / bound | caught |
|:--|--:|:--|
| 1 ulp | 0.0003 to 0.03 | no: below what rounding does anyway |
| 1e-10 | 230 to 24,000 | yes, at every length and position |

The same two perturbations were made by hand in the library, in `twiddleTable`
(plan.go, used by the Stockham, iterative and Rader/Bluestein engines and by
the float32 plans): `t[1]` raised by one ulp, then by 1e-10. The 1 ulp mutation
left `TestAccuracyBound` green; the 1e-10 one turned it red, 1,891 failures
across all ten families, default and FMA.

So the bound detects a twiddle that is wrong by more than rounding noise and not
one that is wrong by an ulp. Equality with the Go reference (the exact mode,
[#121](https://github.com/go-fft/fft/issues/121) step 2) is what guards
bit-level behaviour.
