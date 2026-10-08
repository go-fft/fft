#!/bin/sh
# phase3: composites r24/hsw/c2 x i/s/s1 (TestHswComp, r29b), pow2 s vs s1.
cd ~/r29 && . ./gate.sh
for r in 6,1000 1001,4000 4001,9000 9001,16384; do
  gated comp3.txt ./r29b.test -test.run '^TestHswComp$' -hsw.comp=5,40,${r}
done
gated ab-s1.txt ./r29b.test -test.run '^TestIntelAB$' -intel.ab=7,40 -intel.set="32=a:cur,s:cur,s1:cur;64=a:cur,s:cur,s1:cur;512=a:cur,s:cur,s1:cur,s1:8x4x16;1024=a:cur,s:cur,s1:cur;2048=a:cur,s:cur,s1:cur;4096=a:cur,s:cur,s1:cur;8192=a:cur,s:cur,s1:cur,s1:8x8x4x4x8;16384=a:cur,s:cur,s1:cur;32768=a:cur,s:cur,s1:cur;65536=a:cur,s:cur,s1:cur,s1:8x8x8x8x16"
echo "# PHASE3 DONE $(date +%T)" >> comp3.txt
