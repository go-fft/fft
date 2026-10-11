#!/bin/sh
# hwab.sh CORE ROUNDS OUT MAXLOAD ARGS -- V1 V2 ...: interleaved runs of ./V.test ARGS,
# the order rotated every round; each run waits for a 1-minute load below MAXLOAD
# and is rerun (up to 4 times) if the load reached MAXLOAD+1 by its end.
CORE=$1; ROUNDS=$2; OUT=$3; MAX=$4; shift 4
ARGS=""
while [ "$1" != "--" ]; do ARGS="$ARGS $1"; shift; done; shift
cd ~/r37
: > $OUT
n=$#
gate() {
  while :; do
    l=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l=$l -v m=$MAX 'BEGIN{exit !(l<m)}'; then return; fi
    sleep 15
  done
}
for r in $(seq 1 $ROUNDS); do
  i=0
  while [ $i -lt $n ]; do
    k=$(( (i + r) % n + 1 ))
    eval v=\${$k}
    try=0
    while :; do
      gate
      tmp=$OUT.tmp
      echo "# round $r variant $v $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" > $tmp
      GOMAXPROCS=1 taskset -c $CORE ./$v.test $ARGS >> $tmp 2>&1
      le=$(cut -d' ' -f1 /proc/loadavg)
      echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $tmp
      try=$((try+1))
      if awk -v l=$le -v m=$MAX 'BEGIN{exit !(l<m+1)}' || [ $try -ge 4 ]; then cat $tmp >> $OUT; break; fi
      echo "# discarded round $r variant $v end load $le" >> $OUT
    done
    i=$((i+1))
  done
done
rm -f $OUT.tmp
echo "# done $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
