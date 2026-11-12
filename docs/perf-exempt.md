# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/specs_test.go` or is named
here with its reason. The method and the table are in [perf.md](perf.md).

## Plan constructors: plan cost is excluded on both sides

A plan row builds its plan before the timed loop, and scipy.fft's plan cache is
warm after the calibration call, so neither side times planning. The constructors
therefore have no row of their own; scipy.fft has no constructor to compare with
(pocketfft builds a plan inside the first call for a length).

- `NewPlan`: plan construction, excluded on both sides.
- `NewPlanWith`: as `NewPlan`.
- `NewPlan32`: as `NewPlan`.
- `NewPlanN`: as `NewPlan`.
- `NewPlanNWith`: as `NewPlan`.
- `NewPlanN32`: as `NewPlan`.
- `NewRealPlan`: as `NewPlan`.
- `NewRealPlanWith`: as `NewPlan`.
- `NewRealPlan32`: as `NewPlan`.
- `NewRealPlan2`: as `NewPlan`.
- `NewRealPlan2With`: as `NewPlan`.
- `NewRealPlan2_32`: as `NewPlan`.
- `NewDCTPlan`: as `NewPlan`.
- `NewDCTPlan32`: as `NewPlan`.
- `NewDSTPlan`: as `NewPlan`.
- `NewDSTPlan32`: as `NewPlan`.
- `NewMDCTPlan`: as `NewPlan`.
- `NewMDCTPlan32`: as `NewPlan`.

## Accessors

- `Plan.Len`: returns a stored field.
- `Plan32.Len`: as `Plan.Len`.
- `PlanN.Len`: as `Plan.Len`.
- `PlanN.Shape`: copies a short slice of lengths; no reference.
- `PlanN32.Len`: as `Plan.Len`.
- `PlanN32.Shape`: as `PlanN.Shape`.
- `RealPlan.Len`: as `Plan.Len`.
- `RealPlan32.Len`: as `Plan.Len`.
- `RealPlan2.SpectrumLen`: as `Plan.Len`.
- `RealPlan2_32.SpectrumLen`: as `Plan.Len`.
- `MDCTPlan.Len`: as `Plan.Len`.
- `MDCTPlan32.Len`: as `Plan.Len`.
- `DCTPlan.Len`: as `Plan.Len`.
- `DCTPlan.Type`: as `Plan.Len`.
- `DCTPlan32.Len`: as `Plan.Len`.
- `DCTPlan32.Type`: as `Plan.Len`.
- `DSTPlan.Len`: as `Plan.Len`.
- `DSTPlan.Type`: as `Plan.Len`.
- `DSTPlan32.Len`: as `Plan.Len`.
- `DSTPlan32.Type`: as `Plan.Len`.

## Transforms with no SciPy or NumPy counterpart

SciPy and NumPy have no MDCT. The oracle is the O(N²) definition (`mdct_test.go`),
the cost is `BenchmarkMDCT`, and the transform is a fold plus the DCT-IV, which has
its own rows. FFTW has no MDCT either: the comparison is FFTW's `REDFT11` (the
DCT-IV) of `N/2` points plus the same fold and unfold, single thread, Apple M4 Max,
1-min load 4.5 (2026-11-12), float64, ns per call, MDCT / IMDCT:

| N (samples) | go-fft | FFTW REDFT11 + fold |
|--:|:--|:--|
| 128 | 138 / 134 | 90 / 90 |
| 256 | 231 / 227 | 166 / 168 |
| 2048 | 1,727 / 1,698 | 1,670 / 1,660 |

At the AAC long block (2,048 samples, 1,024 coefficients) go-fft is within 4 % of
the FFTW-built transform; at 128 and 256 it is 1.4 to 1.5 times slower, which is
the small-size DCT-IV, tracked in [#132](https://github.com/go-fft/fft/issues/132).

- `MDCTPlan.MDCT`: no reference tool; see above.
- `MDCTPlan.IMDCT`: as `MDCTPlan.MDCT`.
- `MDCTPlan32.MDCT`: as `MDCTPlan.MDCT`.
- `MDCTPlan32.IMDCT`: as `MDCTPlan.MDCT`.
- `SineWindow`: a closed form computed once per block length and kept by the caller, so no timed row; `scipy.signal.windows.cosine` is the same window (`sin(π(n+1/2)/M)`), and is the reference if a row is ever added.
- `KBDWindow`: computed once per block length and kept by the caller, like `SineWindow`; `scipy.signal.windows.kaiser_bessel_derived` is its counterpart, and the one-time cost is not a row.
