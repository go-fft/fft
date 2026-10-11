# Round 40 raw data (2026-10-11)

Neoverse-N1 (cfarm424, FFTW 3.3.10 + numpy 2.5.3 in `~/gofft-bench`), used by
this round alone, one pinned core (`GOMAXPROCS=1 taskset -c 40`), binaries
cross-compiled with go1.27.1 from the v0.26.1 tag (and v0.25.0 for the A/B),
stripped, gzipped, their sha256 checked on the host (`scripts/r40-SHA256SUMS`).
Every timed run waited for a one-minute load below 1.5. Findings: `REPORT.md`.

- `parity/r1`, `parity/r2`: `benchmarks/remote/run.sh` (as
  `scripts/run-remote-pinned.sh`: cds to its own directory, adds the `-1`
  suffix) run twice back to back by `scripts/r40parity.sh`, each with its
  `REPORT.md`, `go_bench.txt`, `go_plan.txt`, `fftw.json`, `ref.json` and
  `parity.log`; `parity-runs.log` gives the load at each start and end.
- `parity/partab-v0.26.1.txt`: `../current-numbers-v0.26.0/partab.go.txt` on
  r1 and r2 (FFTW = median of the two runs), the command line first.
  `partab-v0.26.1-fftw6.txt`: the same with FFTW the median of six runs
  (Round 37's four and these two), as a cross-check.
- `ab/ab-v025-v0261-9.txt` (+ `.ratios.txt`): the external harness
  (`scripts/hw_place_test.go.txt`, Round 37's, unchanged; module file
  `scripts/harness-go.mod.txt`, its `replace` pointing at a checkout of the
  tag) built against v0.25.0 and v0.26.1, `-test.run=^TestHwPlace$
  -hw.place=11,200,4 -hw.sizes=C1009,C10007,C1024,C1080`, nine rounds, binary
  order rotated (`scripts/r40ab.sh`, Round 37's `hwab.sh` with its directory
  changed). Ratios by Round 37's `scripts/znan/main.go.txt`.
- `ab/bench-v025-v0261-9.txt` (+ `.ratios.txt`): each tag's
  `benchmarks` test binary, `-test.bench='^Benchmark(Complex|FFT2Plan)_GoFFT$/^(1009|10007|1024|256x256|512x512)$'
  -test.benchtime=1s -test.count=1`, nine rotated rounds, same driver.
