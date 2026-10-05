#!/bin/sh
# Round 21, N1 batch C: the block-split prototype against the interleaved passes, one core.
cd ~/gofft-bench/r21 || exit 1
R=${1:-7}
: > split.txt
for r in $(seq 1 $R); do
  echo "round $r $(cat /proc/loadavg)" >> split.txt
  GOMAXPROCS=1 taskset -c 40 ./split.n1 -test.run '^$' -test.bench 'BenchmarkSplit' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> split.txt
done
echo "end $(cat /proc/loadavg)" >> split.txt
echo DONE >> split.txt
