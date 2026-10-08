#!/bin/sh
# abn.sh ARCH CORE ROUNDS OUT BENCH V1 V2 ...: interleaved A/B/n of ./V_ARCH.test, the order rotated every round.
A=$1; CORE=$2; ROUNDS=$3; OUT=$4; BENCH=$5; shift 5
cd ~/r30/ab
: > $OUT
n=$#
for r in $(seq 1 $ROUNDS); do
  i=0
  while [ $i -lt $n ]; do
    k=$(( (i + r) % n + 1 ))
    eval v=\${$k}
    echo "# round $r variant $v $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
    GOMAXPROCS=1 taskset -c $CORE ./${v}_$A.test -test.run xxx -test.bench "$BENCH" -test.benchtime ${BT:-0.2s} >> $OUT 2>&1
    i=$((i+1))
  done
done
echo "# done $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
