#!/bin/sh
# Round 21, N1 batch F: fifteen interleaved rounds of the rows batch E left within their spread, and the threshold for complex128 and complex64 2-D plans.
cd ~/gofft-bench/r21/f || exit 1
R=${1:-15}
: > ab15.txt; : > ab15all.txt; : > thr.txt
for r in $(seq 1 $R); do
  if [ $((r % 2)) = 1 ]; then O="main br"; else O="br main"; fi
  echo "round $r $(cat /proc/loadavg)" >> ab15.txt
  for v in $O; do
    GOMAXPROCS=1 taskset -c 40 ./bench-$v -test.run '^$' -test.bench '^BenchmarkAB$/^(C|R)$/^(1000|1080|1296|1920|2000|6000|1009|10007|262144)$' -test.benchtime 300ms -test.count 1 | sed -n "s/^Benchmark/r=$r v=$v Benchmark/p" >> ab15.txt
  done
  echo "round $r $(cat /proc/loadavg)" >> ab15all.txt
  for v in $O; do
    ./bench-$v -test.run '^$' -test.bench '^BenchmarkAB$/^2D$/^(64|256|512|1024)$' -test.benchtime 300ms -test.count 1 | sed -n "s/^Benchmark/r=$r v=$v Benchmark/p" >> ab15all.txt
  done
done
for r in $(seq 1 7); do
  echo "round $r $(cat /proc/loadavg)" >> thr.txt
  ./bench-br -test.run '^$' -test.bench 'R21Threshold' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> thr.txt
done
echo "end $(cat /proc/loadavg)" >> thr.txt
echo DONE >> thr.txt
