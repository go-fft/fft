# Round 40 — Neoverse-N1 parity at v0.26.1 (2026-10-11)

Measurement only; no library code changed.

**Why.** v0.26.0 moved Bluestein's three pointwise products to
`kernels.CMulTo` on every architecture. On arm64 that is the same scalar loop
behind one more call (no kernel). The N1 column of "Current numbers" came from
Round 37's runs on the branch that became v0.24.0, so complex 10,007
(Bluestein) and 1,009 (Rader) had not been measured against FFTW since.

## Host and method

- cfarm424 (Neoverse-N1), used by this round alone; FFTW 3.3.10 built from
  source with NEON, numpy 2.5.3, scipy 1.18.1 (pocketfft) in `~/gofft-bench`.
- Binaries cross-compiled with go1.27.1 from the `v0.26.1` tag (365a1e3),
  `-trimpath -ldflags=-s`, gzipped; the sha256 of each was checked on the host
  before it was unpacked (`scripts/r40-SHA256SUMS`). The parity package's
  `bench.arm64` is byte-identical to `bench-v0.26.1.test` of the A/B below.
- `benchmarks/remote/run.sh` patched to cd to its own directory and to add the
  `-1` suffix `report.py` expects when `taskset` pins one core
  (`scripts/run-remote-pinned.sh`; the same two changes as Round 37's copy).
- `GOMAXPROCS=1 taskset -c 40`, two full runs back to back
  (`scripts/r40parity.sh`), each gated on a one-minute load below 1.5.
- Table: `current-numbers-v0.26.0/partab.go.txt` (unchanged): go-fft = mean
  of the two runs' medians of three repetitions, FFTW = median of the two
  runs' `fftw_ns`.

Load (host clock, UTC): r1 started 16:16:35 at 0.04 and ended at 1.07; r2
started at 1.07 and ended at 1.09 — one pinned core kept busy is a load of
about 1 on this host. Both runs passed the correctness gate (24/24 against
`numpy.fft`, `rtol=1e-9`).

## Parity (`parity/partab-v0.26.1.txt`)

go-fft time ÷ FFTW time, below 1 means go-fft is faster; "docs" is the N1
column of "Current numbers (v0.26.0)" (Round 37, v0.24.0 branch):

| transform | go-fft ns (r1 / r2) | FFTW ns (r1 / r2) | go/FFTW | docs |
|:--|--:|--:|--:|--:|
| complex 256 | 1,012 / 1,012 | 1,149 / 1,149 | 0.881 | 0.88 |
| complex 1,024 | 4,923 / 4,875 | 6,335 / 6,319 | 0.774 | 0.77 |
| complex 4,096 | 23,890 / 24,124 | 41,033 / 40,867 | 0.586 | 0.58 |
| complex 65,536 | 732,183 / 720,751 | 1,232,651 / 1,271,714 | 0.580 | 0.58 |
| complex 1,048,576 | 22,026,784 / 20,306,061 | 44,140,574 / 45,968,755 | 0.470 | 0.47 |
| complex 1,000 | 6,013 / 6,140 | 7,180 / 7,156 | 0.848 | 0.84 |
| complex 1,080 | 6,901 / 6,867 | 7,667 / 7,668 | 0.898 | 0.89 |
| complex 1,920 | 12,014 / 11,973 | 13,588 / 13,493 | 0.886 | 0.88 |
| complex 1,009 (Rader) | 21,684 / 21,589 | 47,979 / 48,302 | **0.449** | 0.45 |
| complex 1,296 | 8,778 / 8,725 | 11,336 / 11,403 | 0.770 | 0.77 |
| complex 10,007 (Bluestein) | 419,691 / 411,281 | 586,057 / 620,476 | **0.689** | 0.72 |
| RFFT 256 | 716 / 721 | 722 / 719 | 0.998 | 0.991 |
| RFFT 1,024 | 2,952 / 2,929 | 4,205 / 4,202 | 0.699 | 0.69 |
| RFFT 4,096 | 14,236 / 14,391 | 19,845 / 19,883 | 0.721 | 0.71 |
| RFFT 65,536 | 390,605 / 394,015 | 545,857 / 542,864 | 0.721 | 0.72 |
| RFFT 1,048,576 | 10,710,002 / 10,227,502 | 18,481,928 / 19,187,894 | 0.556 | 0.56 |
| RFFT 1,000 | 3,743 / 3,738 | 3,957 / 3,977 | 0.943 | 0.94 |
| RFFT 1,080 | 4,166 / 4,200 | 4,206 / 4,209 | 0.994 | 0.995 |
| RFFT 1,920 | 6,842 / 6,759 | 7,361 / 7,313 | 0.927 | 0.93 |
| 2-D 64×64 | 24,754 / 24,316 | 30,946 / 31,076 | 0.791 | 0.80 |
| 2-D 128×128 | 129,783 / 126,366 | 196,271 / 193,680 | 0.657 | 0.66 |
| 2-D 256×256 | 700,287 / 668,170 | 1,273,538 / 1,271,900 | 0.538 | 0.50 |
| 2-D 512×512 | 3,380,578 / 3,211,345 | 6,834,860 / 6,873,959 | 0.481 | 0.47 |
| 2-D 1024×1024 | 18,630,553 / 18,066,327 | 43,784,148 / 44,445,458 | 0.416 | 0.41 |

**Counts: 24 of 24 at or above FFTW, faster on all 24**, unchanged from the
docs. Against numpy.fft and scipy.fft: faster on all 24 rows in both runs
(go-fft takes 0.08–0.68 of numpy's time and 0.09–0.86 of scipy's). Against
gonum: faster on all 19 1-D rows in both runs. The closest rows are again
RFFT 256 (0.998) and RFFT 1,080 (0.994).

Cross-check (`parity/partab-v0.26.1-fftw6.txt`): with FFTW the median of six
runs (Round 37's four and these two), every row stays below 1; complex 10,007
is 0.696 and RFFT 256 0.997.

## Does CMulTo's extra call cost anything on arm64?

**No; complex 10,007 is 2% faster on v0.26.1.** One external harness (Round
37's `hw_place_test.go`, exported API only, four heap placements per row,
`scripts/hw_place_test.go.txt`), built against v0.25.0 and against v0.26.1,
nine rounds with the binary order rotated (`scripts/r40ab.sh`, Round 37's
`hwab.sh`), `-hw.place=11,200,4` (`ab/ab-v025-v0261-9.txt`, `.ratios.txt`;
ratios by Round 32's `znan`), v0.26.1 ÷ v0.25.0, median of the rounds
(range):

| row | v0.25.0 ns | v0.26.1 ns | v0.26.1 ÷ v0.25.0 |
|:--|--:|--:|:--|
| complex 10,007 (Bluestein, changed) | 420,429 | 411,509 | **0.980** (0.972–0.988) |
| complex 1,009 (Rader, no code changed) | 21,767 | 21,625 | 0.993 (0.990–1.000) |
| complex 1,024 (control) | 4,905 | 4,917 | 1.002 (0.993–1.016) |
| complex 1,080 (control) | 6,902 | 6,917 | 1.002 (0.996–1.006) |

The best placement of each binary gives the same: 0.977, 0.993, 0.999, 0.999.
Every round of 10,007 is below 1. `CMulTo(dst, a, b)` replaces two
`for j … a[j] = src[j]*w[j]` loops and one in-place `kernels.CMul` with a
three-operand loop that reslices its operands to `n` first; the bounds checks
that lets the compiler drop are a plausible cause, not a measured one. Rader's
1,009 runs no code that changed between the tags (among the files built for
arm64, `git diff v0.25.0 v0.26.1` touches only `bluestein.go`, `plann.go`,
`comp_route_arm64.go` and `internal/kernels/{cmulto,cmul_dispatch}.go`); its
0.7% is the binary, inside the controls' spread.

A second instrument, each tag's benchmarks package itself
(`Benchmark{Complex,FFT2Plan}_GoFFT`, `-test.benchtime=1s -test.count=1`,
nine rotated rounds, `ab/bench-v025-v0261-9.txt`), agrees and is noisier:
10,007 0.988 (0.953–1.013), 1,009 0.999, 1,024 0.995, 2-D 256² 1.003
(0.904–1.155), 2-D 512² 1.010.

## Anything odd

- **2-D 256×256 moved from 0.50 to 0.54** (go-fft 684 µs here, 620–649 µs in
  every repetition of Round 37's four runs). It is not code: arm64's 2-D path
  is the same between v0.24.0 and v0.26.1 except that `contiguousLines` now
  takes its scratch buffer lazily (`sk.zn` is always nil on arm64), and in the
  benchmark-binary A/B one binary ran 2-D 256² anywhere from 610 to 705 µs
  between processes (v0.25.0 610–692, v0.26.1 619–705), a spread whose upper
  half the two parity runs happened to sample (700 and 668 µs). Like Cascade
  Lake's 2-D 512² (Round 35 §5) it looks like allocation placement; this
  round did not prove that.
- **FFTW's 10,007 moved 5.9% between the two runs** (586 and 620 µs); Round
  37's four runs read 583–604 µs. Against FFTW's fastest of the six runs,
  go-fft's 10,007 is 0.709.
- The harness A/B started while the load was still falling from the second
  parity run (0.60 at its first process); every process started below 1.5,
  none ended above 1.21, and nothing was discarded or rerun.
- The first attempt at the benchmark-binary A/B matched no benchmark (an
  alternation that crossed the `/` on which Go splits `-test.bench`); its
  empty output was discarded and the run redone with one pattern per level.
