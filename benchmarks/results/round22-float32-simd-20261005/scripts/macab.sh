#!/bin/zsh
# Round 22 arm64 A/B on this Mac; each run waits for 1-min load < 3.
cd ${0:a:h}/bin
waitload() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); if (( l < 3.0 )); then echo "load $(sysctl -n vm.loadavg)"; return; fi; sleep 20; done; }
B='^Benchmark(Complex|Real|CReal)(32)?_GoFFT$'
F='^BenchmarkF32ND$/^(PlanN-\[256_256\]|RealPlan2-256x256|DCT2-4096)-f(32|64)$'
for r in 1 2 3 4 5; do
  waitload; echo "== pass round $r"
  GOMAXPROCS=1 ./ftB.arm64 -test.run '^$' -test.bench 'SK32Pass' -test.benchtime 0.1s
done > ../data/m4-pass.txt 2>&1
for r in 1 2 3 4 5; do
  if (( r % 2 )); then order=(A B); else order=(B A); fi
  for x in $order; do
    waitload; echo "== e2e round $r bench$x"
    GOMAXPROCS=1 ./bench$x.arm64 -test.run '^$' -test.bench "$B" -test.benchtime 0.3s
    waitload; echo "== nd round $r ft$x"
    GOMAXPROCS=1 ./ft$x.arm64 -test.run '^$' -test.bench "$F" -test.benchtime 0.3s -test.cpu 1
  done
done > ../data/m4-e2e.txt 2>&1
echo DONE >> ../data/m4-e2e.txt
