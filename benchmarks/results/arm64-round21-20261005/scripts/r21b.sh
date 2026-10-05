#!/bin/sh
# Round 21, N1 batch B: fan-out rules (all cores), strip widths and 2-D parts (one core), gap grid with the order rotated each round.
cd ~/gofft-bench/r21 || exit 1
B=./r21b.n1
R=${1:-7}
: > ft.txt; : > sw.txt; : > parts-b.txt; : > gap-b.txt
for r in $(seq 1 $R); do
  echo "round $r $(cat /proc/loadavg)" >> ft.txt
  $B -test.run '^$' -test.bench '^BenchmarkFT$' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> ft.txt
  echo "round $r $(cat /proc/loadavg)" >> sw.txt
  GOMAXPROCS=1 taskset -c 40 $B -test.run '^$' -test.bench '^BenchmarkSW$' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> sw.txt
  echo "round $r $(cat /proc/loadavg)" >> parts-b.txt
  GOMAXPROCS=1 taskset -c 40 $B -test.run '^$' -test.bench 'R21Parts2D' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> parts-b.txt
  echo "round $r $(cat /proc/loadavg)" >> gap-b.txt
  case $((r % 3)) in 0) G="576 2112 4032";; 1) G="2112 4032 576";; 2) G="4032 576 2112";; esac
  for g in $G; do
    GAP=$g GOMAXPROCS=1 taskset -c 40 $B -test.run '^$' -test.bench 'GapSK' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r g=$g Benchmark/p" >> gap-b.txt
  done
done
echo "end $(cat /proc/loadavg)" >> gap-b.txt
echo DONE >> gap-b.txt
