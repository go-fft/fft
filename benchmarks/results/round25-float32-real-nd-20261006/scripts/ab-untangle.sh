#!/bin/sh
# Round 25 amd64: main (A) against the branch (B), five interleaved rounds,
# order alternating, one pinned core; then the branch-only R25 benchmarks.
cd ~/r25
CORE=${CORE:-40}
B='^Benchmark(Real|CReal)(32)?_GoFFT$'
F='^BenchmarkF32ND$/^(PlanN-\[64_64\]|PlanN-\[256_256\]|PlanN-\[1024_1024\]|RealPlan2-256x256|RealPlan2-1024x1024)-f(32|64)$'
for r in 1 2 3 4 5; do
  if [ $((r % 2)) = 1 ]; then order="A B"; else order="B A"; fi
  for x in $order; do
    echo "== e2e round $r bench$x"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./bench$x.amd64 -test.run '^$' -test.bench "$B" -test.benchtime 0.3s
    echo "== nd round $r ft$x"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./ft$x.amd64 -test.run '^$' -test.bench "$F" -test.benchtime 0.3s -test.cpu 1
  done
  echo "== r25 round $r ftB"; uptime
  GOMAXPROCS=1 taskset -c $CORE ./ftB.amd64 -test.run '^$' -test.bench 'R25' -test.benchtime 0.2s -test.cpu 1
done > ab.log 2>&1
echo DONE >> ab.log
