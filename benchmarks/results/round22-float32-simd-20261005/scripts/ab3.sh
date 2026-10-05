#!/bin/sh
# Round 22 amd64, final: shipped routing (odd-first for float32) vs main.
cd ~/f32simd/r2
while pgrep -f ab2.sh > /dev/null; do sleep 10; done
CORE=40
./ftC.amd64 -test.count 1 > testC.log 2>&1; echo "exit $?" >> testC.log
B='^Benchmark(Complex|Real|CReal)(32)?_GoFFT$'
F='^BenchmarkF32ND$/^(PlanN-\[256_256\]|RealPlan2-256x256|DCT2-4096)-f(32|64)$'
for r in 1 2 3 4 5; do
  if [ $((r % 2)) = 1 ]; then order="A C"; else order="C A"; fi
  for x in $order; do
    echo "== e2e round $r bench$x"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./bench$x.amd64 -test.run '^$' -test.bench "$B" -test.benchtime 0.3s
    echo "== nd round $r ft$x"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./ft$x.amd64 -test.run '^$' -test.bench "$F" -test.benchtime 0.3s -test.cpu 1
  done
done > e2eC.log 2>&1
echo DONE >> e2eC.log
