#!/bin/sh
# Interleaved: same source (v0.2.0), built by go1.26.4 and go1.27.1.
cd ~/gofft-bench/cmp || exit 1
out=results.txt; : > $out
for r in 1 2 3 4 5; do
  for v in 1.26.4 1.27.1; do
    GOMAXPROCS=1 taskset -c 3 ./bench-go$v -test.run '^$' -test.bench 'Benchmark(Complex|Real|CReal|FFT2Plan)_GoFFT' -test.benchtime 300ms -test.count 1 \
      | sed -n "s/^Benchmark/go$v Benchmark/p" >> $out
  done
done
echo DONE >> $out
