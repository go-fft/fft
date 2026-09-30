# Fresh parity runs, 2026-09-30

Each directory holds one complete run of `benchmarks/run.sh` on the go-fft
code of the "Round 3" section of BENCHMARKS.md, taken just BEFORE that
section's multicore fan-out fix: the generated report
(`REPORT.md`) and the four raw inputs it was built from (`go_bench.txt`,
`go_plan.txt`, `fftw.json`, `ref.json`). The correctness gate (every go-fft
transform against `numpy.fft`, rtol 1e-9) passed 24/24 on each host.

| directory | host | FFTW |
|:--|:--|:--|
| `arm64-apple-m4-max-20260930` | Apple M4 Max, macOS, the local workstation (under background load from other work, load average 3.4–5) | Homebrew 3.3.11 bottle |
| `amd64-zen3-epyc7773x-20260930` | AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420, load average ≈ 2–4 on 128 threads | 3.3.10 from source, SSE2/AVX/AVX2/FMA |
| `arm64-neoverse-n1-20260930` | ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424, load average < 0.4 | 3.3.10 from source, NEON |

On the two Linux hosts Go was not installed at a usable version, so the Go
benchmark and verify binaries were cross-compiled (go1.26.4) from the same
commit and run there; the C harness and the Python references ran natively
(numpy 2.5.3, scipy 1.18.1, pyfftw 0.15.1 in a venv). Everything is
single-threaded except go-fft's 2-D rows, which use its multicore path
(GOMAXPROCS = all hardware threads of each host).

**The 2-D rows predate the fan-out fix** and show what it fixed: on the 64- and
128-thread hosts, small 2-D shapes were split into one goroutine per line.
128×128 ran at 8.5× FFTW on Zen 3 and 3.4× on Neoverse-N1. The fix is
measured separately in BENCHMARKS.md ("Round 3", multicore fan-out): 128×128 is
2.69× faster on Zen 3 and 1.39× faster on N1. The 1-D rows are unaffected by it.
