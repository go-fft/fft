# Speed against scipy.fft

The comparison is with scipy.fft 1.18.1 (pocketfft, one worker), measured by
[refbench](https://github.com/go-scikit/refbench) v0.1.2 from `bench/`. The
generated table is [perf-table.md](perf-table.md) (data: `perf.json`); the public
functions without a row are in [perf-exempt.md](perf-exempt.md). The earlier
multi-host report is [../BENCHMARKS.md](../BENCHMARKS.md) (FFTW, numpy.fft and
gonum columns, dated rounds); the optimisation narrative before it is
[perf-history.md](perf-history.md).

## Method

- **One core on both sides.** refbench sets `GOMAXPROCS=1` around the run and
  starts the Python worker with `OMP/OPENBLAS/MKL/VECLIB_*_THREADS=1`. go-fft
  reads `GOMAXPROCS` when a transform runs (`parWorkerCount` in `parallel.go`,
  fixed by go-fft/fft#107; before v0.28.16 it was read once at package
  initialisation), so `-cpu 1`, the environment or `runtime.GOMAXPROCS(1)` all
  give a one-core run; `bench/perf_test.go` refuses to run on more than one core.
  (On several cores the N-D rows allocate 26 to 36 objects per call for
  goroutine spawns instead of 0 to 1.)
- **Same transform, same data, checked first.** Every row builds its input from
  `frac(i*k) - 0.5` (k the golden ratio's and the plastic number's fractional
  parts; real part, imaginary part), identically in Go and in the NumPy
  expression the reference runs, and the two real-inverse and Hermitian rows use
  a half spectrum whose first and last bins are real. Two tests need `uv` and run by
  hand: `go test -run TestDataParity -perf.parity` compares the SHA-256 of the
  Go arrays with NumPy's (all equal on 2026-11-08), and `go test -run
  TestValueParity -perf.parity` compares the values `scipy.fft` returns for the
  same data with the Go result: the largest difference is at most 1e-15 of the
  largest value in float64 rows (about 3.5e-16 at 1000 and 1009 points, tolerance
  1e-9) and 2.1e-7 in float32 rows (tolerance 2e-4), for all 141 rows. The only
  row without a value check is `NextFastLen`, whose rule differs from scipy's
  on purpose (its doc comment).
- **Plan and setup cost is excluded on both sides.** A `Plan`, `RealPlan`,
  `PlanN`, `DCTPlan`... row builds its plan before the timed loop and passes a
  destination slice; the reference is the one-shot scipy.fft call, whose
  plan comes from pocketfft's plan cache after the calibration call. A one-shot
  row (`fft.FFT`, `scipy.fft.fft`) finds the plan in each library's cache
  (go-fft memoizes plans by length, `plan.go`), so it is not counted either. The
  plan constructors have no row (perf-exempt.md). scipy allocates its result in
  every row; the `Plan` rows write into a caller slice, which is the point of the
  API and is what their allocation count of 0 records.
- **Sizes.** 1-D: 1024 (power of two), 1000 (5-smooth) and 1009 (prime) for the
  complex and real transforms and their plans, 1024 for the rest; 2-D: 256x256;
  N-D: 32x32x32; DCT/DST type 2, orthonormal, at 1024 points and 128x128; float32
  rows against scipy's float32 (complex64) transforms. The `With` rows use
  `Options{Norm: NormOrtho}` against `norm='ortho'`.
- **Not scipy.fft.** The window rows are against NumPy's `hanning`, `hamming`,
  `blackman`, `bartlett` and scipy.signal's `blackmanharris`; `PSD` is against
  `scipy.signal.periodogram(detrend=False)`; `Spectrogram` against the same
  computation done by `scipy.fft.rfft` over a `sliding_window_view` of frames
  (scipy.signal.spectrogram scales by the window power, Spectrogram does not).
- **Interleaved rounds, load gate.** Rounds alternate Go-first and scipy-first;
  refbench refuses to time at a 1-minute load of 8 or more, and the table says
  `deferred` when no window opened.

```
cd bench
GOMAXPROCS=1 go test -run TestPerf -cpu 1 .                 # CI: allocations, table, inventory
GOMAXPROCS=1 go test -run TestPerf -cpu 1 -perf.run .       # timing (writes perf.json and the table)
GOMAXPROCS=1 go test -run TestPerf -cpu 1 -perf.record .    # allocation counts only
```

## Status (2026-11-08)

The committed `perf.json` is **deferred**: allocation counts only, the host's
1-minute load was 26 when the table was generated. No ratio in this repository
is a verdict until the first `-perf.run` at a load below 8 replaces it.

What the counts show, without timing:

- The `Plan`, `RealPlan`, `RealPlan2`, `PlanN`, `DCTPlan` and `DSTPlan` rows
  allocate nothing, in float64 and float32, at 1000 and 1024 points, 256x256.
- The one-shot 2-D and N-D rows allocate 1 to 3 times (float32 2-D and N-D: 3;
  `RFFTN`, `IRFFTN`: 6 to 8), the 1-D ones once.
- Filed from these counts, and fixed in v0.28.16: go-fft/fft#107 (GOMAXPROCS
  read at init), go-fft/fft#108 (the odd-length real plan allocated 2 per call at
  1009, now 0; `RFFT` 3 to 1; `FFTReal` 2 to 1, output bit-identical) and
  go-fft/fft#109 (`Spectrogram`: 262 allocations per call at 127 frames, now 4).
  Since go-fft/fft#122, `FFTReal` at an even length runs the real transform and
  mirrors it (about 1.5x faster than promote-then-FFT at 1024 and 4096 points; the
  bits differ in the last place, within the accuracy bound); `FFTRealWith` with
  `Options{Exact: true}` keeps promote-then-FFT. Still open: go-fft/fft#116, odd
  real lengths still run a full complex transform.
