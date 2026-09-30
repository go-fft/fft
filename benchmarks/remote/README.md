# Running the parity sweep on a Linux host without Go

How the Zen 3 and Neoverse-N1 runs in `../results/` were produced, on GCC
Compile Farm hosts with no root and a Go too old to build go-fft.

1. On the host, `setup.sh <FFTW SIMD flags>` builds FFTW 3.3.10 from source
   into `~/gofft-bench/fftw` (the tarball is checked against its SHA-256) and a
   venv with numpy, scipy and pyfftw in `~/gofft-bench/venv`:

       sh setup.sh --enable-sse2 --enable-avx --enable-avx2 --enable-fma   # x86-64
       sh setup.sh --enable-neon                                           # arm64

2. Locally, cross-compile the Go side from the commit being measured and lay
   out the package `run.sh` expects:

       cd benchmarks
       GOWORK=off GOOS=linux GOARCH=amd64 go build -o ../runpkg/verify.amd64 ./verify
       GOWORK=off GOOS=linux GOARCH=amd64 go test -c -o ../runpkg/bench.amd64 .
       mkdir -p ../runpkg/benchmarks
       cp -r cbench fft_reference.py verify_correctness.py report.py ../runpkg/benchmarks/
       cp remote/run.sh ../runpkg/run-remote.sh

3. Copy `runpkg/` to `~/gofft-bench/` on the host and run it:

       FFTW_DESC="3.3.10 built from source with ..." \
           sh ~/gofft-bench/runpkg/run-remote.sh amd64 "<machine description>"

   It repeats `../run.sh` step by step (correctness gate first) and writes
   `~/gofft-bench/runpkg/BENCHMARKS.md`, the report to keep with the raw
   `go_bench.txt`, `go_plan.txt`, `fftw.json` and `ref.json`.
