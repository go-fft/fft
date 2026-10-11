# Round 37 raw data (2026-10-11)

Haswell (cfarm13, Xeon E5-2620 v3, AVX2 without AVX-512, no FFTW, no numpy)
and Neoverse-N1 (cfarm424, FFTW 3.3.10 + numpy 2.5.3 in `~/gofft-bench`),
each used by this round alone, one pinned core (`GOMAXPROCS=1 taskset -c 10`
on Haswell, `-c 40` on N1). Binaries cross-compiled with go1.27.1.
Every timed run waited for a one-minute load below 2 (Haswell) or 1.5 (N1)
and was to be rerun if it reached that plus 1 by its end (`scripts/hwab.sh`;
none was); each file gives the load at the start and end of every run.
Ratios are computed by Round 32's `znan` (`scripts/znan/main.go.txt`) and
Round 33's `partab` (`../current-numbers-v0.23.0/partab.go.txt`).

## haswell/

- `releases-9.txt`: one external harness (`scripts/hwrel/hw_place_test.go.txt`,
  Round 32's `TestZnPlace` with the flags renamed, exported API only) built
  against v0.19.1, v0.20.0, v0.21.0 and v0.23.1, four heap placements per row,
  nine rounds, binary order rotated. `.vs-v0.21.0.txt`, `.vs-v0.19.1.txt`:
  the ratios.
- `untangle-call-15.txt`: `internal/kernels` `TestHwUntangleCall`, the
  Untangle/Retangle wrappers (512-bit kernel called with quads = 0, then the
  AVX2 kernel) against the AVX2 kernel alone, fifteen rounds, one process.
- `rows-9.txt`, `rows-m-11.txt`: `TestHwRows`, the last axis alone with each
  row plan, nine / eleven rotated rounds.
- `whole-e11-15.txt`: `TestHwWhole` (whole N-D transforms, 1-D row plan
  against 4·8·4·8, fifteen rounds; 1024×512 and 256×512 untouched controls)
  and `TestR26AB` on 6144, 10240 (new rule against the old factorization),
  12288 (untouched).
- `e11-sweep-5.txt`, `e11-ab-15.txt`: `TestHwE11`, every candidate for 6144,
  10240, 12288, five rounds; the finalists, fifteen.
- `accuracy.txt`: `TestHwExport` on Haswell, numpy on N1 on the same bytes
  (`scripts/hw_acc.py`), then with a byte planted.
- `mutants.txt`, `mutations-list.txt`: `scripts/hwmut` with
  `scripts/hwrun13.sh` (each mutant built and run on Haswell).
- `suites.txt`: both packages' suites on Haswell.

## n1/

- `untangle-zip-yz-9.txt`, `untangle-x-9.txt`: the kernel variants in one
  process (main, Z, Y, X; see BENCHMARKS.md). The X run's output was copied
  from the terminal of the session (its variant code no longer exists: X is
  now the kernel).
- `ab-main-branch-9.txt` (+ `.ratios.txt`): the external harness against
  v0.23.1 and the branch, nine rounds.
- `parity/`: `benchmarks/remote/run.sh` (Round 33's `run-remote-pinned.sh`,
  which cds to its own directory) main1, br1, br2, main2, the packages'
  sha256 checked on the host (`scripts/hwparity.sh`); `partab-main.txt`,
  `partab-branch.txt`.
- `parts-branch-9.txt`: `TestHwParts` on the branch.
- `accuracy.txt`, `mutants.txt`, `mutations-list.txt`, `suites.txt`.

`scripts/untangle512-tailjump-unverified.patch.txt` is a fix for the
quads = 0 call that was written and NOT shipped: it changes the AVX-512
kernels, which no host of this round could run.
