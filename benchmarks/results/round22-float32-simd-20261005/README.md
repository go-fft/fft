# Round 22: float32 Stockham passes on AVX2 and NEON (2026-10-05)

The measurements behind Round 22 of BENCHMARKS.md.

- `zen3/`: AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420, one core pinned
  with `taskset -c 40`, `GOMAXPROCS=1`, load average 2.4–4.4 on 128 threads.
  - `baseline-main-f32-vs-f64.txt`: main before the change, float32 against
    float64 (`scripts/base.sh`, `scripts/f32vf64.py`).
  - `pass.txt`: every float32 pass a kernel runs, Go pass against kernel,
    five rounds (`BenchmarkSK32Pass`, `scripts/ab1.sh`, `scripts/pass.py`,
    `scripts/passsum.py`).
  - `order.txt`: Plan32's engine with the kernels in pocketfft's and in the
    odd-first radix order (`scripts/round22_order_test.go.txt`,
    `scripts/order.py`).
  - `e2e-final.txt`: main against the shipped branch, five interleaved
    rounds, the 1-D parity rows and three `BenchmarkF32ND` rows
    (`scripts/ab3.sh`, `scripts/ab.py`). `e2e-pocketfft-order-nd.txt` is the
    same comparison before Plan32 took the odd-first order
    (`scripts/ab2.sh`).
  - `test-suite-final.txt`: the package's whole test suite on that machine,
    AVX2 kernels on.
- `m4/`: Apple M4 Max, this workstation, shared; one core (macOS cannot pin),
  each round started only when the 1-minute load average was below 3.
  - `pass.txt`, `e2e.txt`, `order.txt`: as above (`scripts/macab.sh` for
    the first two pass rounds, `scripts/macab2.sh` for the rest).

The mutation scripts (`scripts/mutate_amd64.py`, `scripts/mutate_arm64.py`)
rewrite one line of the generator, regenerate, and run the float32 kernel
tests; every mutation listed in Round 22 failed them. They expect `R22ROOT`
to name a directory holding the fft clone (`fft/`) and a Go build cache
(`gocache/`). FFTW single precision was not measured: neither host has
`libfftw3f`.
