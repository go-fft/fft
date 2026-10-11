#!/bin/sh
# hwq13.sh: Round 37's Haswell in-process runs, queued after the release A/B.
cd ~/r37
while ps -eo pid,args | grep -q "[h]wab.sh 10 9 rel-9"; do sleep 30; done
V=s:cur,s:4x8x4x8,s:4x4x8x8,s:4x8x8x4,s:4x16x16,s:4x4x4x16,s:16x8x8,s:8x4x4x8,s:8x8x4x4
for s in 1024x1024 512x1024 4x256x1024; do
  ./hwab.sh 10 1 rows-$s.txt 2 -test.run '^TestHwRows$' -hw.rows 9,4 -hw.shapes $s -hw.rowvars $V -- br-amd64
done
./hwab.sh 10 1 e11-sweep.txt 2 -test.run '^TestHwE11$' -hw.e11 5,40 -- br-amd64
echo "# queue done $(date +%T)" >> hwq13.log
