# Apple M4 Max against FFTW, go-fft v0.28.9 (2026-11-02)

First parity run of the M4 against FFTW. Summary in BENCHMARKS.md ("Apple M4 Max").

**FFTW control, taken before any timing** (`nm-g-libfftw3.3.dylib.txt`, `nm -g`
on `~/.pkgx/fftw.org/v3.3.11/lib/libfftw3.3.dylib`, the double-precision library
go-fft is compared against): 751 symbols containing `_fftw_`, of which **207
contain `_neon`**. The pkgx bottle is FFTW 3.3.11, rebuilt with NEON
(go-pkgx/packages#399); the cfarm columns use 3.3.10 built from source. The C
harness links `lib/libfftw3.a` of the same bottle.

Host: Apple M4 Max (16 cores), macOS 27.2, go1.27.3, numpy 2.5.3, scipy 1.18.1,
pyfftw 0.15.1 (uv venv, Python 3.13, under `~/Library/Application Support/gofft-bench`).
macOS has no `taskset`: GOMAXPROCS=1, the core is not pinned. The machine is
shared; `scripts/m4run.go.txt` (a cobra program) waits for a one-minute load
below 8 before each pass and re-reads it before each stage.

| directory | what |
|:--|:--|
| `out1`, `out2` | passes 1 and 2 (go-fft = mean of the two) |
| `out1b` | repeat of pass 1 at the end, drift control |
| `out1-loaded1`, `out1-loaded2` | pass 1 tries excluded: load 23.4 and 14.2 |
| `partab-main.txt` | the table (`partab -go out1,out2 -fftw out1,out2,out1b`) |
| `partab-1-1b.txt`, `partab-b2.txt` | pass 1 against its repeat; repeat with pass 2 |
| `m4run.log` | the passes with their load samples |

Pass 1's Go stage began at a load of 7.9 (falling from 14); the other kept
passes ran at 3.2–6.3. Correctness 24/24 in each kept pass. The runs are
interleaved only at pass level (the cfarm procedure), not row by row.
