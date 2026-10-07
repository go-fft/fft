#!/bin/sh
# rot.sh CORE ROUNDS OUT VARIANT... : interleaved runs of $BENCH, the variant
# order rotated by one each round, one pinned core (GOMAXPROCS=1 taskset).
CORE=$1; ROUNDS=$2; OUT=$3; shift 3
: > $OUT
n=$#
for r in $(seq 1 $ROUNDS); do
  k=$(( (r - 1) % n ))
  set -- "$@"
  i=0; ORDER=""
  for v in "$@"; do ORDER="$ORDER $v"; done
  # rotate: drop the first k into the tail
  j=0; ROT=""; TAIL=""
  for v in $ORDER; do if [ $j -lt $k ]; then TAIL="$TAIL $v"; else ROT="$ROT $v"; fi; j=$((j+1)); done
  for v in $ROT $TAIL; do
    echo "# round $r variant $v $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
    GOMAXPROCS=${GMP:-1} taskset -c $CORE ./$v -test.run xxx -test.bench "$BENCH" -test.benchtime ${BT:-0.3s} >> $OUT 2>&1
  done
done
echo "# done $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
