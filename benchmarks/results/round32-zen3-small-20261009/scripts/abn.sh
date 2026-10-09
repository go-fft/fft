#!/bin/sh
# abn.sh CORE ROUNDS OUT ARGS -- V1 V2 ...: interleaved runs of ./V.test ARGS, the order rotated every round.
CORE=$1; ROUNDS=$2; OUT=$3; shift 3
ARGS=""
while [ "$1" != "--" ]; do ARGS="$ARGS $1"; shift; done; shift
cd ~/r32
: > $OUT
n=$#
for r in $(seq 1 $ROUNDS); do
  i=0
  while [ $i -lt $n ]; do
    k=$(( (i + r) % n + 1 ))
    eval v=\${$k}
    echo "# round $r variant $v $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
    GOMAXPROCS=1 taskset -c $CORE ./$v.test $ARGS >> $OUT 2>&1
    i=$((i+1))
  done
done
echo "# done $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
