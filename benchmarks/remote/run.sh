#!/bin/sh
# Reproduces benchmarks/run.sh on a host without Go: the Go binaries are
# cross-compiled from the same commit. Usage: run-remote.sh <arch> "<machine>"
# (copied into runpkg/ as run-remote.sh; see README.md)
set -e
A=$1; MACHINE=$2
GOV=${GOV:-go1.26.4} # the toolchain that cross-compiled the binaries
cd ~/gofft-bench/runpkg/benchmarks
PY=~/gofft-bench/venv/bin/python
F=~/gofft-bench/fftw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "==> [1/5] correctness gate"
../verify.$A | $PY verify_correctness.py
echo "==> [2/5] Go benchmarks"
../bench.$A -test.run='^$' -test.bench='Benchmark(Complex|Real|CReal|FFT2)_' -test.benchtime=1s -test.count=3 | tee go_bench.txt | tail -3
echo "==> [3/5] plan cost"
../bench.$A -test.run='^$' -test.bench='BenchmarkComplexPlan_' -test.benchtime=300ms -test.count=1 > go_plan.txt
echo "==> [4/5] native FFTW"
cc -O3 -I$F/include cbench/fftw_bench.c $F/lib/libfftw3.a -lm -o fftw_bench
./fftw_bench > fftw.json
echo "==> [5/5] numpy/scipy"
$PY fft_reference.py > ref.json
sed -i "s|\"machine\": \".*\"|\"machine\": \"$MACHINE\"|; s|\"go\": \".*\"|\"go\": \"$GOV linux/$A (cross-compiled)\"|" report.py
FFTW_DESC="${FFTW_DESC:-FFTW built by setup.sh}" $PY report.py
echo done
