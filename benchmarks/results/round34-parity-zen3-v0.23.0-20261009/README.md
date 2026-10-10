# Round 34 — Zen 3 parity at v0.23.0 (2026-10-09)

Two parity runs of the v0.23.0 tag (75854d1) on cfarm420 (AMD EPYC 7773X),
back to back, one pinned core (`taskset -c 40`, so GOMAXPROCS is 1 by
affinity), with the harness of `benchmarks/remote/run.sh` patched to run in its
own directory (as Round 32 did; the stock script changes to a fixed
`~/gofft-bench/runpkg`). The binaries were cross-compiled from the tag with
go1.27.1 and checked by SHA-256 on the host; the benchmark binary links the
Round 32 trampolines (`znTwoPassAVX2`), so it is v0.23.0's code.

Why: the Round 32 parity runs, from which the first v0.23.0 tables were
computed, ran at load 3.3–4.2 and read complex 4096 at 1.50 the FFTW time in
one run (20 µs on all three repetitions); a clean run reads it at 0.91.

| run | start (UTC) | load at start | end | load at end | correctness |
|:--|:--|:--|:--|:--|:--|
| run1 | 18:48:34 | 2.18 | 18:58:29 | 3.21 | 24/24 |
| run2 | 18:58:29 | 3.21 | 19:08:14 | 3.67 | 24/24 |

The load includes this round's own pinned core (1.0) and another user's
long-running job pinned to core 3 at nice 15.

Result (computed by `../current-numbers-v0.23.0/partab.go.txt`, output in
`../current-numbers-v0.23.0/zen3.txt`): within 5% of FFTW or faster on 15 of
24 rows (faster on 11); faster than numpy.fft and scipy.fft on 24 of 24 in both
runs; faster than gonum on 19 of 19.
