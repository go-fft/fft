#!/bin/sh
# decomp.sh BIN CORE ROUNDS OUT : R24Decomp rounds, load logged before each.
BIN=$1; CORE=$2; ROUNDS=$3; OUT=$4
: > $OUT
for r in $(seq 1 $ROUNDS); do
  echo "# round $r $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
  GOMAXPROCS=1 taskset -c $CORE ./$BIN -test.run xxx -test.bench R24Decomp -test.benchtime 0.25s >> $OUT 2>&1
done
echo "# done $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
