#!/bin/sh
# usage: ab.sh OUT ROUNDS BENCHTIME PATTERN BIN...   (env CORE, PROCS)
cd ~/r20 || exit 1
OUT=$1; R=$2; BT=$3; PAT=$4; shift 4
C=${CORE:-3}
: > $OUT
echo "START $(uptime)" >> $OUT
i=1
while [ $i -le $R ]; do
  for b in "$@"; do
    if [ -n "$PROCS" ]; then
      GOMAXPROCS=$PROCS ./$b -test.run '^$' -test.bench "$PAT" -test.benchtime $BT -test.count 1 2>&1 | sed -n "s/^Benchmark/$b Benchmark/p" >> $OUT
    else
      GOMAXPROCS=1 taskset -c $C ./$b -test.run '^$' -test.bench "$PAT" -test.benchtime $BT -test.count 1 2>&1 | sed -n "s/^Benchmark/$b Benchmark/p" >> $OUT
    fi
  done
  echo "ROUND $i $(uptime)" >> $OUT
  i=$((i+1))
done
echo DONE >> $OUT
