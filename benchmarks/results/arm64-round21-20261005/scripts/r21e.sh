#!/bin/sh
# Round 21, N1 batch E: interleaved A/B main vs branch (one core: every BenchmarkAB row; all cores: the 2-D rows), then the gap grid with the final kernels.
cd ~/gofft-bench/r21/e || exit 1
R=${1:-7}
: > ab1.txt; : > aball.txt; : > gap-e.txt
for r in $(seq 1 $R); do
  if [ $((r % 2)) = 1 ]; then O="main br"; else O="br main"; fi
  echo "round $r $(cat /proc/loadavg)" >> ab1.txt
  for v in $O; do
    GOMAXPROCS=1 taskset -c 40 ./bench-$v -test.run '^$' -test.bench '^BenchmarkAB$' -test.benchtime 300ms -test.count 1 | sed -n "s/^Benchmark/r=$r v=$v Benchmark/p" >> ab1.txt
  done
  echo "round $r $(cat /proc/loadavg)" >> aball.txt
  for v in $O; do
    ./bench-$v -test.run '^$' -test.bench '^BenchmarkAB$/^2D$' -test.benchtime 300ms -test.count 1 | sed -n "s/^Benchmark/r=$r v=$v Benchmark/p" >> aball.txt
  done
done
for r in $(seq 1 $R); do
  echo "round $r $(cat /proc/loadavg)" >> gap-e.txt
  case $((r % 4)) in 0) G="576 1088 2112 4032";; 1) G="1088 2112 4032 576";; 2) G="2112 4032 576 1088";; 3) G="4032 576 1088 2112";; esac
  for g in $G; do
    GAP=$g GOMAXPROCS=1 taskset -c 40 ./gap.n1 -test.run '^$' -test.bench 'GapSK' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r g=$g Benchmark/p" >> gap-e.txt
  done
done
echo "end $(cat /proc/loadavg)" >> gap-e.txt
echo DONE >> gap-e.txt
