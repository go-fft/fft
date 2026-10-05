#!/bin/sh
# Interleaved A/B: ab.sh ROUNDS BENCHREGEX OUTFILE CPU BENCHTIME BIN_A BIN_B
# Each round runs BIN_A then BIN_B (alternating which goes first), one pinned
# core, GOMAXPROCS=1, and records the load average before each binary.
R=$1; B=$2; OUT=$3; CPU=$4; BT=$5; A=$6; Bb=$7
: > "$OUT"
i=1
while [ "$i" -le "$R" ]; do
  if [ $((i % 2)) -eq 1 ]; then order="$A $Bb"; else order="$Bb $A"; fi
  for v in $order; do
    echo "=== round $i bin $v load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
    GOMAXPROCS=1 taskset -c "$CPU" ./"$v" -test.run XXX -test.bench "$B" -test.benchtime "$BT" -test.count 1 >> "$OUT" 2>&1
  done
  i=$((i + 1))
done
echo "=== DONE load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
