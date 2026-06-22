#!/usr/bin/env bash
# Reproduce the full go-fft performance-parity sweep and regenerate
# ../BENCHMARKS.md. Runs four implementations on the SAME host, SAME inputs,
# SAME sizes, single-threaded, with plan reuse:
#
#   1. go-fft  + gonum  via `go test -bench`        (Go)
#   2. FFTW             via a native C harness      (gold standard)
#   3. numpy.fft + scipy.fft (pocketfft)            (Python)
#   4. correctness gate: go-fft vs numpy.fft        (run FIRST; aborts on mismatch)
#
# Prereqs (macOS):
#   brew install fftw
#   python3 -m venv .venv && .venv/bin/pip install numpy scipy pyfftw
# Set PY to that interpreter, e.g.  PY=.venv/bin/python ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
FFTW_PREFIX="$(brew --prefix fftw 2>/dev/null || echo /opt/homebrew)"
export GOWORK=off
# Single-thread pin for the reference libraries.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "==> [1/5] correctness gate: go-fft vs numpy.fft"
go run ./verify | "$PY" verify_correctness.py

echo "==> [2/5] Go benchmarks: go-fft + gonum (steady state, best-of-3)"
go test -run='^$' -bench='Benchmark(Complex|Real|FFT2)_' -benchtime=1s -count=3 . \
    | tee go_bench.txt

echo "==> [3/5] Go benchmarks: plan/setup cost"
go test -run='^$' -bench='BenchmarkComplexPlan_' -benchtime=300ms -count=1 . \
    | tee go_plan.txt

echo "==> [4/5] native FFTW (C, gold standard)"
cc -O3 -I"$FFTW_PREFIX/include" cbench/fftw_bench.c \
   -L"$FFTW_PREFIX/lib" -lfftw3 -lm -o /tmp/go_fft_fftw_bench
/tmp/go_fft_fftw_bench >fftw.json

echo "==> [5/5] numpy.fft + scipy.fft (pocketfft)"
"$PY" fft_reference.py >ref.json

echo "==> consolidating -> ../BENCHMARKS.md"
"$PY" report.py
echo "done."
