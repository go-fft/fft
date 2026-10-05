#!/bin/sh
# Fifteen interleaved rounds of the rows whose 5-round change was inside the spread.
cd ~/gofft-bench/neon/e2e || exit 1
out=${1:-ab15-1296.txt}; : > $out
uptime >> $out
for r in $(seq 1 15); do
  for v in main neon; do
    for pat in 'BenchmarkComplex_GoFFT/^1296$'; do
      GOMAXPROCS=1 taskset -c 40 ./bench-$v -test.run '^$' -test.bench "$pat" -test.benchtime 500ms -test.count 1 \
        | sed -n "s/^Benchmark/$v Benchmark/p" >> $out
    done
  done
done
uptime >> $out
echo DONE >> $out
