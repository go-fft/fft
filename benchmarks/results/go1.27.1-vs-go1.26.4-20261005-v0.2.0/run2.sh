#!/bin/sh
cd ~/gofft-bench/cmp || exit 1
out=results2.txt; : > $out
for r in $(seq 1 15); do
  for v in 1.26.4 1.27.1; do
    for pat in 'BenchmarkComplex_GoFFT/^(4096|1000|1048576)$' 'BenchmarkReal_GoFFT/^(1080|4096)$' 'BenchmarkCReal_GoFFT/^(256|65536)$'; do
      GOMAXPROCS=1 taskset -c 5 ./bench-go$v -test.run '^$' -test.bench "$pat" -test.benchtime 500ms -test.count 1 \
        | sed -n "s/^Benchmark/go$v Benchmark/p" >> $out
    done
  done
done
echo DONE >> $out
