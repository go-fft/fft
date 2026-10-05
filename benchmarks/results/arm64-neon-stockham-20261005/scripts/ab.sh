#!/bin/sh
# Interleaved A/B: main vs perf-arm64-neon, every go-fft row of the parity harness.
cd ~/gofft-bench/neon/e2e || exit 1
R=${1:-5}; out=${2:-ab.txt}; : > $out
uptime >> $out
for r in $(seq 1 $R); do
  for v in main neon; do
    GOMAXPROCS=1 taskset -c 40 ./bench-$v -test.run '^$' -test.bench 'Benchmark(Complex|Real|CReal|FFT2Plan)_GoFFT' -test.benchtime 300ms -test.count 1 \
      | sed -n "s/^Benchmark/$v Benchmark/p" >> $out
  done
done
uptime >> $out
echo DONE >> $out
