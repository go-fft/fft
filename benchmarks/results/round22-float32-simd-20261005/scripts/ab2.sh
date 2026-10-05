#!/bin/sh
# Round 22 amd64, second run: full test suite with AVX2, radix order, end to end incl. float32 N-D.
cd ~/f32simd/r2
CORE=40
./ord.amd64 -test.count 1 > test.log 2>&1; echo "exit $?" >> test.log
for r in 1 2 3 4 5; do echo "== order round $r"; uptime; GOMAXPROCS=1 taskset -c $CORE ./ord.amd64 -test.run '^$' -test.bench 'R22Order' -test.benchtime 0.2s; done > order.log 2>&1
B='^Benchmark(Complex|Real|CReal)(32)?_GoFFT$'
F='^BenchmarkF32ND$/^(PlanN-\[256_256\]|RealPlan2-256x256|DCT2-4096)-f(32|64)$'
for r in 1 2 3 4 5; do
  if [ $((r % 2)) = 1 ]; then order="A B"; else order="B A"; fi
  for x in $order; do
    echo "== e2e round $r bench$x"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./bench$x.amd64 -test.run '^$' -test.bench "$B" -test.benchtime 0.3s
    if [ $x = A ]; then f=ftA.amd64; else f=ord.amd64; fi
    echo "== nd round $r ft$x"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./$f -test.run '^$' -test.bench "$F" -test.benchtime 0.3s -test.cpu 1
  done
done > e2e.log 2>&1
echo DONE >> e2e.log
