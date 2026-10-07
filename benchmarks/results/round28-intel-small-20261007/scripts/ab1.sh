#!/bin/sh
# ab1.sh BIN OUT ROUNDS,MS : TestIntelAB over the Round 28 candidate set, one pinned core.
BIN=$1; OUT=$2; RM=$3
SET="64=w:cur,z:8x8,z:4x16,z:4x4x4;128=w:cur,w:16x8,w:8x16,z:4x4x8,z:8x4x4,z:4x8x4,z:8x16,z:16x8;256=w:cur,w:16x16,z:cur,z:8x4x8,z:4x8x8;512=w:cur,z:cur,z:8x4x16;1024=w:cur,z:cur,z:8x4x4x8,z:4x8x4x8;2048=w:cur,z:cur,z:8x4x8x8,z:8x8x4x8;4096=w:cur,z:cur,z:8x4x8x4x4;8192=w:cur,z:cur,z:8x8x4x4x8,z:4x8x4x8x8;16384=w:cur,z:cur,z:8x8x4x8x8;32768=w:cur,z:cur,z:4x4x8x8x8x4,z:8x8x4x4x8x4"
{
echo "# start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
GOMAXPROCS=1 taskset -c 3 ./$BIN -test.run '^TestIntelAB$' -intel.ab=$RM -intel.set="$SET"
echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
} > $OUT 2>&1
