#!/bin/sh
# Round 21, N1 batch G: the PlanN threshold (parThresholdN) for complex128 PlanN and RealPlan2, all cores, seven rounds.
cd ~/gofft-bench/r21/g || exit 1
: > thr3.txt
for r in $(seq 1 7); do
  echo "round $r $(cat /proc/loadavg)" >> thr3.txt
  ./bench-br3 -test.run '^$' -test.bench 'R21Threshold' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> thr3.txt
done
echo "end $(cat /proc/loadavg)" >> thr3.txt
echo DONE >> thr3.txt
