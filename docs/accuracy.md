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

`c` is **measured**, not derived: it is twice the worst ratio seen on random
inputs uniform in `[-1, 1)`. It sits below Higham's deterministic worst-case
constant for a radix-2 FFT with exactly rounded twiddles (about 2.8·ε·log2 n),
which random inputs do not approach. The bound is a regression gate, a way to
see that a change did not make the transforms less accurate, and not a proof
that no input exceeds it.

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
  closed-form roots (`N = 8, 12`) and the identities `|w| = 1` and
  `w[2k] = w_{N/2}[k]` (each to 1e-30), `math.Sincos` (to 1e-14), a direct
  double-double sum for 13 lengths, the inverse round trip and the DCT/DST
  inverse round trip (each to 1e-29), a pure tone (1e-28), Parseval (1e-13,
  relative, in float64) and a hand-computed DCT-II and the orthogonality of
  the ortho DCT-II (1e-13, float64).

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
| `c2c` | FFT, IFFT, FFTReal (the real transform plus the conjugate mirror at an even length), FFTRealWith, RealPlan.FFTReal, Plan.FFT/IFFT, Plan.FFTNorm/IFFTNorm, FFTWith/IFFTWith with N (padded and truncated) and Norm | 0.74 | 1.5 |
| `r2c` | RFFT, IRFFT, HFFT, IHFFT, RealPlan and its Norm methods, RFFTWith with N and Norm | 0.92 | 2.0 |
| `c2cN` | FFTN, IFFTN, FFT2, IFFT2, PlanN and its Norm methods, FFTNWith with Axes and Norm | 0.30 | 0.75 |
| `r2cN` | RFFTN, IRFFTN, RFFT2, IRFFT2, RealPlan2 and its Norm methods | 0.28 | 0.75 |
| `r2r` | DCT, IDCT, DST, IDST (types I–IV, backward, ortho and forward norm), their plans, the N-D forms | 0.95 | 2.0 |
| `psd` | PSD, Spectrogram (a squared quantity: the ratio is on the power) | 1.03 | 2.25 |
| `c2c/32` | FFT32, IFFT32, Plan32 and its Norm methods, FFT32With, IFFT32With | 0.42 | 1.0 |
| `r2c/32` | RFFT32, IRFFT32, HFFT32, IHFFT32, RealPlan32 and its Norm methods, the five `...32With` forms | 0.51 | 1.25 |
| `c2cN/32` | FFTN32, IFFTN32, FFT2_32, IFFT2_32, PlanN32 and its Norm methods, the four `...With` forms | 0.15 | 0.5 |
| `r2cN/32` | RFFTN32, IRFFTN32, RFFT2_32, IRFFT2_32, RealPlan2_32 and its Norm methods, the four `...With` forms | 0.15 | 0.5 |
| `r2r/32` | the float32 DCT/DST family | 0.86 | 1.75 |
| `mdct` | MDCTPlan.MDCT/IMDCT, `N` samples to `N/2` coefficients | 0.83 | 1.75 |
| `mdct/32` | MDCTPlan32.MDCT/IMDCT | 0.30 | 0.75 |
| `mixed` | MDCTPlan.MDCTMixed/IMDCTMixed, DCTPlan.DCTMixed/IDCTMixed (DCT-IV): float32 in and out, float64 arithmetic | 0.56, 0.84 of the bound | `mdct` or `r2r` c, plus ε32/2 |

The `mixed` methods round once to float32 on output, so the stated bound is the
float64 one plus that rounding, `err ≤ c64·ε64·log2 n + ε32/2` in relative RMS
(`ε32/2 = 2^-24`), and the output is exactly the float32 of the float64
method's output (`TestMixedIsFloat32OfFloat64`). `TestMixedAccuracy` holds the
MDCT (N from 2 to 4,096) and the DCT-IV (n from 2 to 1,009, all three norms,
both directions) to it against the double-double oracle: the worst ratios to the
bound are 0.56 (MDCT) and 0.84 (DCT-IV). On the same inputs the float32 MDCT
plan's worst error is 1.9e-7, three times the float32 rounding alone (6e-8): it
accumulates `log2 N` float32 stages, which is what these methods avoid.


The `mdct` rows are measured by `TestMDCTAccuracy` against the `O(N·N/2)`
definition `X[k] = Σ x[n]·cos(2π/N·(n + 1/2 + N/4)(k + 1/2))` in double-double
(the 200-bit cosine table of the oracle, the argument reduced exactly in
integers), for `N` from 2 to 4,096 (both routes: `N/2` even through the DCT-IV,
`N/2` odd through the `N`-point FFT, which is the worse one, 0.83 against 0.36 in
float64), two seeds, and the scales 1 and `2/(N/2)`. `n` in the bound is `N`.
`TestMDCTPerfectReconstruction` holds the windowed overlap-add round trip (sine
and Kaiser-Bessel-derived windows) to twice that bound.

Every exported transform of the package is covered, by its plain form, its
plan methods (all sixteen `...Norm` ones included) and its `...With` form,
float32 included. Not covered, because they are not transforms: the windows,
`FFTFreq`/`RFFTFreq`, `FFTShift`, `NextFastLen`.

Measured on arm64 (Apple silicon, Go 1.27); the amd64 kernels and the FMA plans
are asserted by the same test in CI. FMA is not less accurate: its worst ratio
in every family is within 0.05 of the default's (`c2c`: 0.64 against 0.60).
float32 ignores `Options.FMA`. Where the CPU has no FMA3 the test still builds
the FMA plans (its Go reference), so every platform checks them.

Lengths: powers of 2 (2 to 4,096), of 3 (to 2,187), of 5 (to 3,125), of 7 (to
2,401), mixed smooth lengths (6 to 2,310, including 1,000 and 1,001), Rader
primes (17, 19, 31, 127, 257, 1,009; 19 is Rader because 18 is 7-smooth),
Bluestein lengths (23, 47, 323, 1,013, 2,018, 2,039), odd real lengths (9, 15, 105, 343, 1,001, 1,009, 1,013, 3,125), 2-D to 4-D
shapes (4×4 to 128×128, 30×31, 127×33, 1,009×3, 8×9×10, 16³, 17×3×5×4), and the
r2r lengths 2 to 1,009.

The worst ratios sit at the shortest lengths (2 to 25), where `log2 n` is
smallest and the error does not shrink with it. Rader and Bluestein lengths
measure 0.2 to 0.6, the same as the smooth ones. On the qemu architectures
(s390x, ppc64le, loong64, riscv64) CI caps the sweep at 130 points, one seed
and 500 points for N-D shapes to fit its timeout; that run keeps 23 and 47
(Bluestein), 17, 19, 31 and 127 (Rader), smooth lengths, odd real lengths,
N-D shapes and the r2r transforms, which `TestAccuracyCoversEveryRoute`
asserts on every architecture.

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
left `TestAccuracyBound` green; the 1e-10 one turned it red, 3,267 failures
across the float64 families (`c2c`, `r2c`, `c2cN`, `r2cN`, `r2r`, `psd`),
default and FMA. No float32 family fails at 1e-10: it is far below float32's
resolution (`ε` is 1.2e-7), so the float32 bound cannot see it; a twiddle
error of the order of 1e-7 would be the float32 equivalent.

So the bound detects a twiddle that is wrong by more than rounding noise and not
one that is wrong by an ulp. Equality with the Go reference (the exact mode,
[#121](https://github.com/go-fft/fft/issues/121) step 2) is what guards
bit-level behaviour.
