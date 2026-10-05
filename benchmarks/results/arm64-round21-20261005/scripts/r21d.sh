#!/bin/sh
# Round 21, N1 batch D: whole 2-D transforms with src, dst and scratch placed, one core.
cd ~/gofft-bench/r21 || exit 1
R=${1:-3}
: > place.txt
for r in $(seq 1 $R); do
  echo "round $r $(cat /proc/loadavg)" >> place.txt
  GOMAXPROCS=1 taskset -c 40 ./r21d.n1 -test.run '^$' -test.bench 'BenchmarkPlace' -test.benchtime 100ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> place.txt
done
echo "end $(cat /proc/loadavg)" >> place.txt
echo DONE >> place.txt
