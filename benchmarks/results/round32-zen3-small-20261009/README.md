# Round 32 — Zen 3, the smallest sizes: complex 64, and the buffers between the passes (2026-10-08/09)

The measurements behind Round 32 of BENCHMARKS.md. Host: Zen 3 (cfarm420,
AMD EPYC 7773X, a 128-thread VM), used by this round alone, one pinned core
(`GOMAXPROCS=1 taskset -c 40`); its 1-minute load from other users (1.9–3.9)
is printed at the head of every round in every file. "main" is v0.21.0
(bb351ec) throughout; the branch binaries are named after the commit stage
that built them (`br`, `br3`, `br6`, `br7`; `br7` is the final code).

| file | what |
|:--|:--|
| `zen3/place-tags-9.txt` | v0.19.1, v0.20.0, v0.21.0 built with the same external harness (`round32_place_test.go`, `TestZnPlace`), each row at 12 heap placements, nine rounds, the order rotated |
| `zen3/over-tags-7.txt` | the same three tags: `TestR26Overhead` (`Plan.FFT`, `skPlan.transform`, the passes alone, a pool round trip) |
| `zen3/stack-zeroed-inproc-11.txt` | dropped: the buffer as a zeroed Go array on the stack, against the pool, one process |
| `zen3/two-inproc-11.txt` | the two-pass frame kernel against the pool, one process (`TestZnTwoPass`, first version) |
| `zen3/orders-frame-9.txt` | every two-pass order of 16 … 128, and 16·16 against 4·8·8 at 256, with the frame kernel (`TestR30Orders`) |
| `zen3/ab-place-9.txt`, `ab-place2-9.txt`, `ab-place3-9.txt`, `ab-place4-9.txt` | main against the branch at four stages (`br`, `br3`, `br6`, `br7`), `TestZnPlace`, nine rounds, rotated, untouched rows last |
| `zen3/ab-r30-9.txt` | main against `br7`, `BenchmarkR30AB` (Round 30's row set), nine rounds, rotated |
| `zen3/ab-untouched-9.txt` | main against `br7` on rows the round does not touch, eight placements |
| `zen3/frame-inproc-11.txt` | `br6`/`br7`'s code in one process, plans built with the frame kernels off and on (`TestZnFrame`) |
| `zen3/fftw-plans-1.txt` | FFTW's `FFTW_MEASURE` plans and best-of-nine times for 32 … 1024 points (`scripts/fftwplan.c`) |
| `zen3/mutants.txt` | fourteen mutations of the guards, the selection and the generated trampoline, each built and run |
| `zen3/accuracy.txt` | numpy on the exact bytes `TestZnExport` wrote, and four planted faults |
| `zen3/suites.txt` | both packages' suites on Zen 3 (branch, coverage instrumented) |
| `zen3/parity/` | four parity runs (main, branch, branch, main), `parity-ratios.txt` |

Scripts: `abn.sh` (n binaries, the order rotated every round), `znan/`
(a Go reader of those files: per-row medians and per-round ratios),
`fftwplan.c`, `zn_accuracy.py`, `parity32.sh`, `fixreport.sh` (Round 30's).
