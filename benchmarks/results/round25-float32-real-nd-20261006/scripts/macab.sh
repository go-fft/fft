#!/bin/zsh
# Round 25 arm64 A/B on the M4 Max: one load check (1-min < 3) at the start of
# each round; a round never starts above it. Gives up after GIVEUP seconds.
cd ${0:a:h}
GIVEUP=${GIVEUP:-5400}
t0=$(date +%s)
waitload() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); if (( l < 3.0 )); then echo "load $(sysctl -n vm.loadavg)"; return 0; fi; if (( $(date +%s) - t0 > GIVEUP )); then echo "GAVE UP: load $(sysctl -n vm.loadavg)"; return 1; fi; sleep 20; done; }
B='^Benchmark(Real|CReal)(32)?_GoFFT$'
F='^BenchmarkF32ND$/^(PlanN-\[64_64\]|PlanN-\[256_256\]|PlanN-\[1024_1024\]|RealPlan2-256x256|RealPlan2-1024x1024)-f(32|64)$'
for r in 1 2 3 4 5; do
  if (( r % 2 )); then order=(A B); else order=(B A); fi
  waitload || break
  for x in $order; do
    echo "== e2e round $r bench$x"
    GOMAXPROCS=1 ./bench$x.arm64 -test.run '^$' -test.bench "$B" -test.benchtime 0.2s
    echo "== nd round $r ft$x"
    GOMAXPROCS=1 ./ft$x.arm64 -test.run '^$' -test.bench "$F" -test.benchtime 0.2s -test.cpu 1
  done
  echo "== r25 round $r ftB"
  GOMAXPROCS=1 ./ftB.arm64 -test.run '^$' -test.bench 'R25' -test.benchtime 0.1s -test.cpu 1
  echo "load-after $(sysctl -n vm.loadavg)"
done > m4-ab.log 2>&1
echo DONE >> m4-ab.log
