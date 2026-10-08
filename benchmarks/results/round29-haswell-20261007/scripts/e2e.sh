#!/bin/sh
# e2e.sh ROUNDS OUT [BENCH]: BenchmarkAB, main.test and br.test alternating
# every round, each run gated on load < 2 and pinned (gate.sh).
cd ~/r29/e2e && . ../gate.sh
R=$1; OUT=$2; B=${3:-'^BenchmarkAB$'}
for r in $(seq 1 $R); do
  if [ $((r % 2)) -eq 1 ]; then ORDER="main.test br.test"; else ORDER="br.test main.test"; fi
  for v in $ORDER; do
    echo "# round $r variant $v" >> $OUT
    gated $OUT ./$v -test.run xxx -test.bench "$B" -test.benchtime ${BT:-0.3s}
  done
done
echo "# E2E DONE $(date +%T)" >> $OUT
