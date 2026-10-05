# Round 19: a radix-16 pass on amd64

Measurements behind BENCHMARKS.md's Round 19. Hosts: AMD EPYC 7773X (Zen 3),
cfarm420, a 128-thread VM, load average 2.4–5.2; Intel Xeon E5-2620 v3 (Haswell),
cfarm13, bare metal, load average 1.1–2.0. Every run is one core pinned with
`taskset`, `GOMAXPROCS=1`, cross-compiled test binaries (go1.27.1).

- `raw/rules-*.txt`: `BenchmarkRadix16Rules` (`round19_bench_test.go`), the
  current factorization (`cur-…`) against every radix-16 candidate.
  `rules-zen3.txt` ran the first kernel (twiddles stored as for the other
  passes) up to 2^20; `rules-zen3-dup.txt` and `rules-hsw.txt` the shipped
  kernel (pre-duplicated twiddles). `scripts/sweep.sh`, `scripts/med.py`.
- `raw/passes-zen3.txt`: `BenchmarkRadix16Passes`, each pass of a
  factorization timed alone (first kernel).
- `raw/ab-dup.txt`: the first kernel (`fft-r16b`) against the pre-duplicated
  one (`fft-r16c`), interleaved, `BenchmarkRadix16Passes`.
- `raw/nd-hsw.txt` and `raw/ab-*-batchcols.txt`: a build with a batched
  radix-16 kernel for the columns of N-D plans (`skBatch16AVX2`, dropped):
  n×n with and without radix-16 columns, and main against that build on
  every `BenchmarkAB` row.
- `raw/ab2-hsw.txt` (five rounds) and `raw/ab2-zen3.txt` (fifteen rounds):
  main (`fft-mains`) against the code of the pull request (`fft-r16f`),
  interleaved, `BenchmarkAB`. `scripts/ab.sh`, `scripts/abmed.py`.
- `zen3-parity-main/`, `zen3-parity-r16/`: the parity harness
  (`remote/run.sh` as `scripts/run-remote.sh`, which adds the `-1` suffix
  report.py needs on a one-core run) for main and for the branch, run back to
  back by `scripts/parity.sh` on one pinned core.
