#!/bin/sh
# ab2.sh BIN OUT ROUNDS,MS : TestIntelAB, the finalists of ab1, one pinned core.
BIN=$1; OUT=$2; RM=$3
SET="128=w:cur,w:16x8,w:8x16;256=z:cur,z:8x4x8,w:cur;1024=z:cur,z:8x4x4x8,w:cur;2048=z:cur,z:8x8x4x8,w:cur;8192=z:cur,z:8x8x4x4x8,z:4x8x4x8x8,w:cur"
{
echo "# start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
GOMAXPROCS=1 taskset -c 3 ./$BIN -test.run '^TestIntelAB$' -intel.ab=$RM -intel.set="$SET"
echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
} > $OUT 2>&1
