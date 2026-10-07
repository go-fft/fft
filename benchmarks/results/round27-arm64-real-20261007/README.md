# Round 27 raw data (2026-10-07)

The measurements behind BENCHMARKS.md's Round 27, all on Neoverse-N1
(cfarm424, 64 cores, used by this round alone), one pinned core
(`GOMAXPROCS=1 taskset -c 40`). Each run file logs the one-minute load before
every run; `scripts/rot.sh` runs the variants interleaved, the order rotated
by one each round. Binaries cross-compiled with go1.27.1.

Variants: `main.test` is main (2d48924, v0.16.1) with the Round 27
benchmarks added; `br.test` the one-step float64 untangle kernel, `br2.test`
two interleaved steps, `br3.test` three (the kept kernel); `old`/`new` and
`old2`/`new2` the branch with the rotating passes off (`R27ROT=0`) and on;
`f12`/`f15`/`f16.test` `scripts/f32ab_test.go.txt` built against v0.12.0,
v0.15.0 and v0.16.1.

`neoverse-n1/`:

- `shares-main.txt`: `BenchmarkR27Real` (RFFT, IRFFT, and their untangle and
  retangle loops alone) and `BenchmarkR27Parts2D` (a 2-D plan whole, its rows,
  its columns as strips and gathered) on main, five rounds;
  `shares-main.medians.txt` (`scripts/med.py`).
- `untangle-{1,2,3}step.txt`: main against each untangle kernel, five rounds;
  `*.ratios.txt` (`scripts/aban.py`: medians, ratio of medians, range of the
  per-round ratios, spread). `untangle-3step.ratios.txt` also holds two steps
  against three.
- `rotate-a.txt`, `rotate-b.txt`: `BenchmarkR27ND`, rows and strips against
  the rotating passes, five rounds (`rotate-b` adds 256², 512², 1024², 64³);
  `rotate-512-15.txt`: 512² over fifteen rounds.
- `float32-releases.txt`: `BenchmarkAB` and `BenchmarkND` of
  `scripts/f32ab_test.go.txt` against the three releases, five rounds;
  `float32-releases.ratios.txt` (`scripts/f32an.py`);
  `float32-complex-15.txt`: the complex float32 rows of v0.15.0 and v0.16.1
  over fifteen rounds.
- `parity-{main,br,main2,br2}/`: `benchmarks/remote/run.sh` pinned with
  `taskset` (Round 24's `run-remote-pinned.sh`, which adds the `-1` suffix
  report.py expects), main and the branch back to back, twice
  (`scripts/parity.sh`); each correct 24/24.

`mutations-arm64.txt`: the seventeen mutations of the untangle generator
(`scripts/mutate.py`), each caught.
`mutations-rotate.txt`: the four mutations of the rotating passes, each caught.
