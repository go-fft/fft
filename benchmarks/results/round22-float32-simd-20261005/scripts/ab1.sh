#!/bin/sh
# Round 22 amd64 A/B: per-pass (Go vs AVX2 kernel, same binary), then end to end (main vs branch), interleaved.
cd ~/f32simd
CORE=40
B='^Benchmark(Complex|Real|CReal)(32)?_GoFFT$'
for r in 1 2 3 4 5; do
  echo "== pass round $r"; uptime
  GOMAXPROCS=1 taskset -c $CORE ./passbench -test.run '^$' -test.bench 'SK32Pass' -test.benchtime 0.1s
done > pass.log 2>&1
for r in 1 2 3 4 5; do
  if [ $((r % 2)) = 1 ]; then order="bench0 bench1"; else order="bench1 bench0"; fi
  for b in $order; do
    echo "== e2e round $r $b"; uptime
    GOMAXPROCS=1 taskset -c $CORE ./$b -test.run '^$' -test.bench "$B" -test.benchtime 0.3s
  done
done > e2e.log 2>&1
echo DONE >> e2e.log
