#!/bin/sh
# sweep.sh ROUNDS BENCHREGEX OUTFILE CPU BENCHTIME BIN: one binary, ROUNDS runs.
R=$1; B=$2; OUT=$3; CPU=$4; BT=$5; BIN=$6
: > "$OUT"
i=1
while [ "$i" -le "$R" ]; do
  echo "=== round $i load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
  GOMAXPROCS=1 taskset -c "$CPU" ./"$BIN" -test.run XXX -test.bench "$B" -test.benchtime "$BT" -test.count 1 >> "$OUT" 2>&1
  i=$((i + 1))
done
echo "=== DONE load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
