# Round 23: the data kept split between AVX2 passes on amd64

Measurements behind BENCHMARKS.md's Round 23. Host: AMD EPYC 7773X (Zen 3),
cfarm420, a 128-thread VM, load average 2.5–3.6 throughout. Every run is one
core pinned with `taskset -c 40`, `GOMAXPROCS=1`, cross-compiled test binaries
(go1.27.1). Haswell (cfarm13) was loaded (16–29) and measured nothing.

- `raw/rules1.txt` (radix-4 split only), `raw/rules2.txt` (radix 4 and 8,
  each candidate also run interleaved, `-split.twins`, to 2^14),
  `raw/rules3.txt` (three sweeps to 2^20): `BenchmarkSplitRules`
  (`round23_bench_amd64_test.go`), main's factorization (`cur-…`) against
  every split candidate (`s-…`). `scripts/sumrules.py` summarizes one.
- `raw/passes1.txt`: `BenchmarkSplitPasses`, each pass of a factorization
  timed alone, interleaved (`i`) and split (`s`), with its layout mode (`m1`
  interleaved in, `m2` split both sides, `m3` interleaved out).
- `raw/ab1.txt`, `raw/ab2.txt`: `TestSplitAB -split.ab=9,30` and `11,30`, the
  best candidates per length in one process, the order rotated every round;
  `ab2.txt` has the chosen table (`s:rule`) and the composite lengths.
- `raw/e2e-first-zen3.txt`, `raw/e2e15-first-zen3.txt`: main against the first
  build (whose `StockhamPassLayout` did not inline), `BenchmarkAB`, five and
  fifteen rounds.
- `raw/e2e-final-zen3.txt`, `raw/e2e15-final-zen3.txt`: main against the code
  of the pull request, five rounds of every row and fifteen of the rows whose
  spread exceeded their ratio. `scripts/ab.sh` (Round 19's), `scripts/abmed.py`,
  `scripts/chain.sh`.
- `raw/mutants-zen3.txt`: nine mutations of the generator and the twiddle table
  (`scripts/mutate.py`), each run against `TestSplitEachPassMatchesScalar` and
  against the whole-transform tests alone.
- `zen3-parity-main/`, `zen3-parity-split/`: the parity harness for main
  (3754264) and the branch, back to back on one pinned core
  (`scripts/parity23.sh`, `scripts/run-remote.sh`); `raw/parity23.out` has the
  loads.
