#!/bin/sh
# Round 25 amd64, N-D: main (ftA) against the branch with strips (ftB3),
# five interleaved rounds, order alternating, one pinned core; then the
# branch's parts and strip-width benchmarks each round.
cd ~/r25
CORE=${CORE:-40}
F='^BenchmarkF32ND$/^(PlanN-\[64_64\]|PlanN-\[256_256\]|PlanN-\[32_32_32\]|PlanN-\[1024_1024\]|RealPlan2-256x256|RealPlan2-1024x1024)-f(32|64)$'
for r in 1 2 3 4 5; do
  if [ $((r % 2)) = 1 ]; then order="A B3"; else order="B3 A"; fi
  for x in $order; do
    echo "== nd round $r ft$x"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./ft$x.amd64 -test.run '^$' -test.bench "$F" -test.benchtime 0.3s -test.cpu 1
  done
  echo "== parts round $r ftB3"; uptime
  GOMAXPROCS=1 taskset -c $CORE ./ftB3.amd64 -test.run '^$' -test.bench 'R25Parts2D32|R25StripWidth32' -test.benchtime 0.2s -test.cpu 1
done > ab2.log 2>&1
echo DONE >> ab2.log
