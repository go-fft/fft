#!/bin/sh
# g.sh OUT cmd... : wait for 1-min load < 1.5, run cmd pinned on core 3, log load at start and end.
OUT=$1; shift
while :; do l=$(cut -d" " -f1 /proc/loadavg); [ $(echo "$l < 1.5" | bc) = 1 ] && break; sleep 20; done
echo "# start $(date +%T) load $(cut -d" " -f1-3 /proc/loadavg) : $*" > $OUT
cd ~/r31 && GOMAXPROCS=1 taskset -c 3 "$@" >> $OUT 2>&1
echo "# end $(date +%T) load $(cut -d" " -f1-3 /proc/loadavg)" >> $OUT
