#!/bin/sh
# gate.sh: sourced. gated OUT CMD...: waits until the 1-minute load is below
# 2, runs CMD pinned to core $CORE with GOMAXPROCS=1, appends to OUT with the
# load before and after; reruns (up to 3 times) a chunk whose end load reached
# 3 (the run's own core counts 1).
CORE=${CORE:-10}
idle() { awk -v l="$(cut -d' ' -f1 /proc/loadavg)" -v m="$1" 'BEGIN{exit !(l<m)}'; }
waitidle() { while ! idle 2; do sleep 20; done; }
gated() {
  out=$1; shift
  for try in 1 2 3 4; do
    waitidle
    echo "# start try $try $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg) :: $*" >> $out
    GOMAXPROCS=1 taskset -c $CORE "$@" >> $out 2>&1
    echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $out
    if idle 3; then return 0; fi
    echo "# DISCARD: load reached 3 during the chunk" >> $out
  done
}
