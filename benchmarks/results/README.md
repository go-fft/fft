# Fresh parity runs

Each directory holds one complete run of `benchmarks/run.sh` (on Linux,
`remote/run.sh`): the generated report (`REPORT.md`) and the four raw inputs it
was built from (`go_bench.txt`, `go_plan.txt`, `fftw.json`, `ref.json`).

## 2026-10-05, Round 20: large transforms on Cascade Lake

`round20-cascade-large-20261005` holds the host probes (clock, latency,
bandwidth, FFTW's own scaling), the pass-by-pass and schedule experiments,
the interleaved A/B runs (Cascade Lake, Zen 3) and two pinned parity runs on
Cascade Lake, `main` and the blocked schedule, each correct 24/24. See its
README.

## 2026-10-05, Round 19: a radix-16 pass on amd64

`round19-radix16-20261005` holds the factorization sweeps, per-pass timings,
interleaved A/B runs and scripts of BENCHMARKS.md's Round 19 (Zen 3, cfarm420;
Haswell, cfarm13), and two pinned single-core parity runs on Zen 3, of main
and of the branch. See its README.

## 2026-10-05, Round 17: small and mid sizes on amd64

`round17-amd64-small-20261005` holds the interleaved A/B runs, sweeps and
scripts of BENCHMARKS.md's Round 17, and two pinned single-core parity runs
(Zen 3, cfarm420; Cascade Lake, cfarm151), each correct 24/24. See its README.

## 2026-10-05, perf-arm64-neon: NEON Stockham passes (Round 18)

`arm64-neon-stockham-20261005` holds the measurements behind Round 18 of
BENCHMARKS.md, on Neoverse-N1 (cfarm424, load average below 1, one pinned
core) and on an Apple M4 Max (shared, load average 3-5, not pinned):

- `*/pass-*.txt`: every NEON-capable pass timed alone, Go pass against kernel
  (`BenchmarkSKPassNEON`, `scripts/pass.sh`, `scripts/passan.py`); `pass-v1`
  and `pass-v2` are the first radix-4 kernel and its `LDP`+`ZIP` variant
  (`zip` rows), `pass-final` the code of the pull request.
- `*/ab-*.txt`: interleaved A/B of main against the branch on every go-fft row
  of the parity harness (`scripts/ab.sh`, `ab15.sh` and `ab15-1296.sh` for the
  fifteen-round reruns, `scripts/aban.py`).
- `*/gap-*.txt`, `apple-m4-max/passgap-*.txt`: the scratch-gap experiment, on
  whole transforms (`scripts/gapexp_test.go.txt`, with `setGap` made a
  variable) and pass by pass (`BenchmarkSKPassNEON` with the scratch placed
  at the gap given in the file name); not shipped.
- `neoverse-n1/parity/`: the parity report of the branch against FFTW
  (`remote/run.sh`, go1.27.1).

## 2026-10-05, v0.2.0: go1.27.1 against go1.26.4

`go1.27.1-vs-go1.26.4-20261005-v0.2.0` is not a parity run: it times the same
source built by the two toolchains, interleaved, on Zen 3 (cfarm420),
Neoverse-N1 (cfarm424) and Cascade Lake (cfarm151). It holds the raw `go test`
lines, the scripts (`run.sh`, `run2.sh` for the fifteen-round rerun) and
`analyze.py`, which prints the per-row ratios (`*.ratios.txt`). See Round 16 of
BENCHMARKS.md.

## 2026-10-04, v0.1.7, Cascade Lake

| directory | host | FFTW |
|:--|:--|:--|
| `amd64-cascadelake-20261004-v0.1.7` | Intel Xeon (Cascade Lake), 8 vCPUs, cfarm151, load average < 1 | 3.3.10 from source, SSE2/AVX/AVX2/AVX-512/FMA |

The released v0.1.7, whose code on this machine (AVX-512, radix 8 throughout)
is that of v0.1.5. The Python references are the same versions as on the other
hosts (numpy 2.5.3, scipy 1.18.1, pyfftw 0.15.1), installed with `uv` on Python
3.13 because the host's default Python is 3.6. Correctness 24/24. Rows at or
above FFTW: 5/24; at or above numpy.fft and scipy.fft: 24/24.

## 2026-10-04, v0.1.5

| directory | host | FFTW |
|:--|:--|:--|
| `amd64-zen3-epyc7773x-20261004-v0.1.5` | AMD EPYC 7773X (Zen 3), cfarm420, load average ≈ 3–4 on 128 threads | 3.3.10 from source, SSE2/AVX/AVX2/FMA |
| `arm64-neoverse-n1-20261004-v0.1.5` | Neoverse-N1, 64 cores, cfarm424, load average < 1 | 3.3.10 from source, NEON |

The released v0.1.5 (Round 10), cross-compiled with go1.26.4. The correctness
gate passed 24/24 on both hosts, and the 2-D go-fft column is the reused
`PlanN` row, which the sweep now selects itself. Rows at or above FFTW: 6/24 on
Zen 3, 7/24 on Neoverse-N1. At or above numpy.fft and scipy.fft: 24/24 on Zen 3;
on Neoverse-N1, 23/24 and 20/24.

## 2026-10-04, after Round 8

| directory | host | FFTW |
|:--|:--|:--|
| `amd64-zen3-epyc7773x-20261004` | AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420, load average ≈ 3 on 128 threads | 3.3.10 from source, SSE2/AVX/AVX2/FMA |
| `arm64-neoverse-n1-20261004` | ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424, load average < 1 | 3.3.10 from source, NEON |

Taken on the code of BENCHMARKS.md's "Round 8" section, cross-compiled with
go1.26.4. The correctness gate passed 24/24 on both hosts. The 2-D go-fft
column is the reused `PlanN` row (`BenchmarkFFT2Plan_GoFFT`), run right after
the sweep with the same settings and appended to `go_bench.txt`. The sweep
scripts of that commit did not yet select it; they now do.

Against the same day's run of v0.1.2 on Zen 3 (go-fft time ÷ FFTW time):
complex 4096 2.15 → 1.53, complex 65536 1.84 → 1.07, complex 2²⁰ 1.56 → 0.59,
RFFT 2²⁰ 1.62 → 1.02, IRFFT 2²⁰ 2.08 → 0.93. Rows at or above FFTW parity:
2/24 → 5/24, and 24/24 against numpy.fft.

## 2026-09-30

Each directory holds one complete run of `benchmarks/run.sh` on the go-fft
code of the "Round 3" section of BENCHMARKS.md, taken just BEFORE that
section's multicore fan-out fix: the generated report
(`REPORT.md`) and the four raw inputs it was built from (`go_bench.txt`,
`go_plan.txt`, `fftw.json`, `ref.json`). The correctness gate (every go-fft
transform against `numpy.fft`, rtol 1e-9) passed 24/24 on each host.

| directory | host | FFTW |
|:--|:--|:--|
| `arm64-apple-m4-max-20260930` | Apple M4 Max, macOS, the local workstation (under background load from other work, load average 3.4–5) | Homebrew 3.3.11 bottle |
| `amd64-zen3-epyc7773x-20260930` | AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420, load average ≈ 2–4 on 128 threads | 3.3.10 from source, SSE2/AVX/AVX2/FMA |
| `arm64-neoverse-n1-20260930` | ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424, load average < 0.4 | 3.3.10 from source, NEON |

On the two Linux hosts Go was not installed at a usable version, so the Go
benchmark and verify binaries were cross-compiled (go1.26.4) from the same
commit and run there; the C harness and the Python references ran natively
(numpy 2.5.3, scipy 1.18.1, pyfftw 0.15.1 in a venv). Everything is
single-threaded except go-fft's 2-D rows, which use its multicore path
(GOMAXPROCS = all hardware threads of each host).

**The 2-D rows predate the fan-out fix** and show what it fixed: on the 64- and
128-thread hosts, small 2-D shapes were split into one goroutine per line.
128×128 ran at 8.5× FFTW on Zen 3 and 3.4× on Neoverse-N1. The fix is
measured separately in BENCHMARKS.md ("Round 3", multicore fan-out): 128×128 is
2.69× faster on Zen 3 and 1.39× faster on N1. The 1-D rows are unaffected by it.

## 2026-10-05, perf-arm64-2: NEON strips, split layout, fan-out threshold (Round 21)

`arm64-round21-20261005` holds the measurements behind Round 21 of
BENCHMARKS.md on Neoverse-N1 (cfarm424): the 2-D decompositions, strip widths,
fan-out rules and thresholds, the split-layout prototype, the scratch-gap grid,
the interleaved A/B runs (`*.ratios.txt`, `*.medians.txt` from the scripts in
`scripts/`), and two parity runs (`neoverse-n1/parity-main`,
`neoverse-n1/parity-br`, main and the branch, each correct 24/24). There is
no Apple M4 Max data: the workstation's load never fell below 3.

## 2026-10-05, float32-simd: float32 Stockham passes on AVX2 and NEON (Round 22)

`round22-float32-simd-20261005` holds the measurements behind Round 22 of
BENCHMARKS.md: main's float32 against float64 on Zen 3 (cfarm420), the float32
passes timed alone (Go against kernel), the radix-order comparison, and the
interleaved end-to-end A/B runs, on Zen 3 (one pinned core) and on an Apple
M4 Max (rounds started at a 1-minute load below 3). See its README.

## 2026-10-06, perf-amd64-split: the data kept split between AVX2 passes (Round 23)

`round23-amd64-split-20261006` holds the measurements behind Round 23 of
BENCHMARKS.md on Zen 3 (cfarm420, one pinned core): the factorization sweeps
with the split layout, each pass timed alone, the rotated in-process A/B of the
candidates, the interleaved end-to-end A/B runs against main, the mutation
runs, and two parity runs (`zen3-parity-main`, `zen3-parity-split`, each
correct 24/24). See its README.

## 2026-10-06, perf-composites: smooth composite lengths (Round 24)

`round24-composites-20261006` holds the measurements behind Round 24 of
BENCHMARKS.md on Cascade Lake (cfarm151) and Neoverse-N1 (cfarm424), one
pinned core: each composite's passes timed alone, the power-of-two tail and
radix-12 sweeps, the interleaved A/B runs against main, and four parity runs
per host (main and the branch, twice, each correct 24/24). See its README.

## 2026-10-06, float32-real-nd: float32 untangle and batched N-D columns (Round 25)

`round25-float32-real-nd-20261006` holds the measurements behind Round 25 of
BENCHMARKS.md, all on Zen 3 (cfarm420, one pinned core unless stated):
- the interleaved A/B runs, main against the branch: RFFT32/IRFFT32 (`zen3/e2e-untangle.txt`) and the N-D rows (`zen3/nd-strips.txt`);
- the parts of a 2-D float32 plan, the untangle alone, the strip-width sweep and its 15-round check, and the fan-out rule on eight cores;
- the test and mutation logs (`zen3/tests-*.txt`).

There is no Apple M4 Max data: the workstation's 1-minute load never fell
below 3.

## 2026-10-07, perf-amd64-comp2: composites and small real sizes on Zen 3 (Round 26)

`round26-amd64-comp2-20261007` holds the measurements behind Round 26 of
BENCHMARKS.md, all on Zen 3 (cfarm420, one pinned core): each pass of the
composite and small power-of-two lengths timed alone, the RFFT split into
its half transform and its untangle, Round 24's Intel rule timed on AMD, the
radix-10/15/20 factorizations and the 148-length sweep behind the AMD rule,
the interleaved A/B runs against main, the generator mutants, and four parity
runs (main and the branch, twice each). See its README.

## 2026-10-07, arm64-real: float64 NEON untangle, rotating N-D passes, float32 on Neoverse-N1 (Round 27)

`round27-arm64-real-20261007` holds the measurements behind Round 27 of
BENCHMARKS.md, all on Neoverse-N1 (cfarm424, one pinned core): the untangle's
share of RFFT and IRFFT and the three untangle kernels against main, the
rotating N-D passes against rows and strips, the float32 rows of v0.12.0,
v0.15.0 and v0.16.1, and the parity runs, main and the branch twice. See its
README.

## 2026-10-07, intel-small: the split layout at 512 bits on Cascade Lake (Round 28)

`round28-intel-small-20261007` holds the measurements behind Round 28 of
BENCHMARKS.md, all on Cascade Lake (cfarm151, one pinned core): the
factorization sweeps in every layout and width, the rotated A/B runs, the
end-to-end runs against main (including the build whose use of Z16..Z31 slowed
the composites), the mutation log and four parity runs. See its README.

`round29-haswell-20261007` holds the measurements behind Round 29 of
BENCHMARKS.md, all on Haswell (cfarm13, one pinned core, each run gated on a
load below 2): the power-of-two sweeps and A/B split against interleaved, the
composite sweeps of Round 24's and Round 26's rules, the hardware counters,
the accuracy check against numpy and the end-to-end runs against main. See its
README.

`round30-small-real-20261008` holds the measurements behind Round 30 of
BENCHMARKS.md, on Zen 3 (cfarm420) and Neoverse-N1 (cfarm424): RFFT and
complex 256 split part by part, the untangle kernel variants, the RealPlan
buffer dropped, N1's power-of-two radix sweep, fifteen-round main/branch A/B
runs, the complex-64 layout investigation, the mutation log and four parity
runs per host. See its README.
