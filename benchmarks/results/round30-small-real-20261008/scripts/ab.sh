#!/bin/sh
# ab.sh ARCH CORE ROUNDS OUT: full suites, then interleaved main/branch A/B of BenchmarkR30AB, order alternating per round.
A=$1; CORE=$2; ROUNDS=$3; OUT=$4
cd ~/r30/ab
echo "# tests $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" > $OUT.tests
GOMAXPROCS=4 ./br_$A.test -test.count=1 >> $OUT.tests 2>&1; echo "br exit $?" >> $OUT.tests
( cd ../kt 2>/dev/null || mkdir -p ../kt; cd ../kt; GOMAXPROCS=4 ../ab/kern_$A.test -test.count=1 ) >> $OUT.tests 2>&1; echo "kern exit $?" >> $OUT.tests
: > $OUT
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) -eq 1 ]; then ORDER="main br"; else ORDER="br main"; fi
  for v in $ORDER; do
    echo "# round $r variant $v $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
    GOMAXPROCS=1 taskset -c $CORE ./${v}_$A.test -test.run xxx -test.bench "BenchmarkR30AB" -test.benchtime 0.2s >> $OUT 2>&1
  done
done
echo "# done $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
