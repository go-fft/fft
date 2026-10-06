# Round 25: float32 untangle and batched N-D columns (2026-10-06)

The measurements behind Round 25 of BENCHMARKS.md.

- `zen3/`: AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420, `GOMAXPROCS=1`, `taskset -c 40` (eight cores, `taskset -c 40-47`, for the fan-out runs), load average 2.3–3.4 on 128 threads.
  - `e2e-untangle.txt`: main (`benchA`/`ftA`) against the branch with the untangle kernels (`benchB`/`ftB`). There are five interleaved rounds, alternating order (`scripts/ab-untangle.sh`), for:
    - the RFFT32/IRFFT32 and float64 parity rows;
    - five `BenchmarkF32ND` rows;
    - each round, `BenchmarkR25Untangle32` (Go loop against kernel) and `BenchmarkR25Parts2D32`.
  - `nd-strips.txt`: main against the branch with the batched columns (`ftB3`), five interleaved rounds (`scripts/ab-strips.sh`). It holds the `BenchmarkF32ND` rows, plus `BenchmarkR25Parts2D32` and `BenchmarkR25StripWidth32` each round.
  - `strip-width-15.txt`: 15 rounds of strip widths 16 and 32.
  - `fanout8.txt`, `fanout8-15.txt`: `BenchmarkR25Fanout32`, the shared fan-out rule against one goroutine, seven and 15 rounds.
  - `tests-untangle-mutants.txt`, `tests-strips-mutants.txt`, `tests-strips-mutants-2.txt`: the new tests, every mutant of the generated assembly, and the whole test suite of both packages, on that machine with the AVX2 kernels on.
- `scripts/`:
  - `abr.py` summarizes an A/B log: medians, the ratio, the pair range, and f32 ÷ f64 on each side.
  - `parts.py` and `fan.py` summarize the branch-only benchmarks.
  - `mutate_*.py` rewrite one instruction of a generated `.s`. On amd64 they build a mutant test binary per mutation; on arm64 they run the tests in place. The arm64 untangle mutations (8 of 8 caught) and batch mutations (6 of 6 caught) ran on the Apple M4 Max; their output is in BENCHMARKS.md.
  - `macab.sh` is the load-gated M4 A/B. It never started a round: the 1-minute load stayed above 3.
  - The Python scripts read `R25ROOT`, a directory holding the clone (`fft/`), the binaries (`bin*/`) and a Go build cache (`gocache/`).
