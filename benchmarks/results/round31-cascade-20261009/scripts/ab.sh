#!/bin/sh
# ab.sh OUT ROUNDS BENCH BT V1 V2 ... : interleaved runs, variant order rotated each round, load gated.
OUT=$1; R=$2; B=$3; BT=$4; shift 4
: > $OUT
cd ~/r31
for r in $(seq 1 $R); do
  while :; do l=$(cut -d" " -f1 /proc/loadavg); [ $(echo "$l < 1.5" | bc) = 1 ] && break; sleep 20; done
  set -- "$@"; k=$(( (r-1) % $# )); i=0; ORDER=""
  for v in "$@"; do ORDER="$ORDER $v"; done
  ROT=$(echo $ORDER | awk -v k=$k "{for(i=0;i<NF;i++){j=(i+k)%NF+1; printf \"%s \", \$j}}")
  for v in $ROT; do
    echo "# round $r variant $v $(date +%T) load $(cut -d" " -f1-3 /proc/loadavg)" >> $OUT
    GOMAXPROCS=1 taskset -c 3 ./$v -test.run xxx -test.bench "$B" -test.benchtime $BT >> $OUT 2>&1
  done
done
echo "# done $(date +%T) load $(cut -d" " -f1-3 /proc/loadavg)" >> $OUT
