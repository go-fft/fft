#!/bin/sh
# gap x NEON grid, interleaved per round. Usage: gap.sh BIN ROUNDS OUT [taskset prefix]
cd "$(dirname "$0")" || exit 1
B=$1; R=$2; out=$3; shift 3; : > $out
uptime >> $out
for r in $(seq 1 $R); do
  for g in ${GAPS:-576 1088 2112 3136}; do
    for ne in ${NEONS:-0 1}; do
      GAP=$g NEON=$ne GOMAXPROCS=1 "$@" ./$B -test.run '^$' -test.bench 'GapSK' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/g=$g neon=$ne Benchmark/p" >> $out
    done
  done
  uptime >> $out
done
echo DONE >> $out
