#!/bin/sh
# ab.sh A B CORE ROUNDS OUT : interleaved A/B of $BENCH, order alternating per round.
A=$1; B=$2; CORE=$3; ROUNDS=$4; OUT=$5
: > $OUT
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) -eq 1 ]; then ORDER="$A $B"; else ORDER="$B $A"; fi
  for v in $ORDER; do
    echo "# round $r variant $v $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
    GOMAXPROCS=1 taskset -c $CORE ./$v -test.run xxx -test.bench "$BENCH" -test.benchtime ${BT:-0.3s} >> $OUT 2>&1
  done
done
echo "# done $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
