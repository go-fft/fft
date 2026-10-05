# Round 20 raw data (2026-10-05)

The measurements behind BENCHMARKS.md's Round 20: why large 1-D transforms
were slow on Cascade Lake (cfarm151, one core pinned with `taskset -c 3`,
`GOMAXPROCS=1`), and the blocked schedule that replaced breadth first there.

- `host/memory-and-clock-151.txt`: `scripts/probe` (Go + assembly,
  cross-compiled): an add chain beside 256- and 512-bit multiplies (clock),
  a pointer chase over a random cycle (latency) and AVX-512/AVX2 streams
  (read, write, copy), on 2 MiB pages advised huge or not.
- `host/fftw-scaling-151.txt`: `scripts/fftw_scale.c` against the host's
  FFTW 3.3.10 (AVX-512): ns per point from 2^10 to 2^20, and FFTW's plans.
- `raw/passes-and-variants-151.txt`: `TestRound20Probe` with
  `R20MODE=passes` (each pass alone, AVX-512 and AVX2, with a Go copy of the
  same shape, `_mem`, which is no floor at small ido: one `copy` per run) and
  `R20MODE=whole` (scratch gaps, page sizes, in place, radix rules, pow2
  kernel), on `main`'s code.
- `raw/blocked-scatter-151.txt`, `raw/blocked-decomp-151.txt`: the first
  prototype (groups, then a Go `copy` scatter), with its parts timed apart
  (`_x1`: the last pass writing dst contiguously, the traffic a fused scatter
  has; `_x2`: the passes over the whole array; `_x3`: the groups).
  Correctness was checked bit for bit in the same run.
- `raw/pair-copy-proto-151.txt`: the first two passes fused through Go
  gathers and scatters, wrong twiddles, timing only.
- `raw/cascade-151.txt`: the shipped groups (kernel scatter), no pair,
  group sizes 4096/8192/16384, AVX-512 (`wtrue`) and AVX2.
- `raw/cascade-pair-151.txt`, `raw/cascade-sweep-151.txt`: with the pair
  (`_ptrue`, `_cN` = chunk width N, `_c0` = no pair), chunk 128 then the
  sweep over chunks 32–256 and group sizes.
- `raw/ab-*-151.txt`: interleaved end-to-end runs (`scripts/ab.sh`,
  summarized by `scripts/ab3.py`), one line per benchmark and binary;
  `ROUND` lines carry the load average.
  - `ab-blocked-151.txt`: groups only (no pair) against `main`.
  - `ab-fin-151.txt`: the pair, threshold 32768, seven rounds,
    `BenchmarkLarge` and `BenchmarkAB`.
  - `ab-fin15-151.txt`: the final code (threshold 65536), fifteen rounds,
    `BenchmarkLarge`, `main2.test` = origin/main at 34c4988 plus the
    benchmark file.
  - `ab-fin-420.txt`, `cascade-420.txt`: Zen 3 (cfarm420), the
    non-regression run and the schedule forced on (AVX2) for reference.
- `parity-main/`, `parity-r20/`: `benchmarks/remote/run.sh` on cfarm151,
  pinned, `main` then this branch, each correct 24/24. The reports were
  regenerated locally after adding the `-1` suffix `report.py` expects.
