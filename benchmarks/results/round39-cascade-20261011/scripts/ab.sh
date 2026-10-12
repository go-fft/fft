#!/bin/sh
# ab.sh OUT ROUNDS ARGS... : run main.test and br.test alternately (order rotated each round),
# each waiting for load < 1.5, pinned on core 3, GOMAXPROCS=1; prefix each output line with round and binary.
OUT=$1; R=$2; shift 2
cd ~/r39
echo "# start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg) main $(sha256sum main.test | cut -c1-16) br $(sha256sum br.test | cut -c1-16) : $*" > $OUT
i=0
while [ $i -lt $R ]; do
  if [ $((i % 2)) = 0 ]; then order="main br"; else order="br main"; fi
  for b in $order; do
    while :; do l=$(cut -d" " -f1 /proc/loadavg); [ $(echo "$l < 1.5" | bc) = 1 ] && break; sleep 20; done
    GOMAXPROCS=1 taskset -c 3 ./$b.test "$@" 2>&1 | grep -v '^PASS' | sed "s/^/R$i $b /" >> $OUT
  done
  i=$((i+1))
done
echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT
