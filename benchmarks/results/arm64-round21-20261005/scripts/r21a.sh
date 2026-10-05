#!/bin/sh
# Round 21, N1 batch A: fan-out floors (all cores), 2-D parts (one core), gap grid (one core).
cd ~/gofft-bench/r21 || exit 1
B=./r21exp.n1
R=${1:-5}
: > fanout.txt; : > parts.txt; : > gap.txt
for r in $(seq 1 $R); do
  echo "round $r $(cat /proc/loadavg)" >> fanout.txt
  $B -test.run '^$' -test.bench 'R21Fanout' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> fanout.txt
  echo "round $r $(cat /proc/loadavg)" >> parts.txt
  GOMAXPROCS=1 taskset -c 40 $B -test.run '^$' -test.bench 'R21Parts2D' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> parts.txt
  echo "round $r $(cat /proc/loadavg)" >> gap.txt
  for g in 576 1088 2112 3136 4032; do
    GAP=$g GOMAXPROCS=1 taskset -c 40 $B -test.run '^$' -test.bench 'GapSK' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r g=$g Benchmark/p" >> gap.txt
  done
done
echo "end $(cat /proc/loadavg)" >> gap.txt
echo DONE >> gap.txt
