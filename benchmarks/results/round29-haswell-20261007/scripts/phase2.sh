#!/bin/sh
# phase2: TestIntelAB on the pow2 finalists, 7 rotated rounds, 40 ms (100 ms from 2^17).
cd ~/r29 && . ./gate.sh
for set in \
 "32=a:cur,s:cur,a:8x4;64=a:cur,s:cur,s:16x4,a:4x16;128=a:cur,s:cur,a:4x4x8,s:4x4x8;256=a:cur,s:cur,a:4x8x8,s:4x8x8" \
 "512=a:cur,s:cur,s:8x4x16,s:4x8x16,a:4x8x16;1024=a:cur,s:cur,s:8x16x8,s:8x4x4x8;2048=a:cur,s:cur,s:16x8x16,s:8x8x4x8" \
 "4096=a:cur,s:cur,s:8x4x4x4x8,s:4x8x4x8x4;8192=a:cur,s:cur,s:8x8x4x4x8,s:8x8x8x4x4,s:4x4x4x4x8x4" \
 "16384=a:cur,s:cur,s:8x4x8x8x8,s:4x8x8x4x4x4;32768=a:cur,s:cur,s:8x8x8x4x16,s:8x8x8x4x4x4" \
 "65536=a:cur,s:cur,s:8x8x8x8x16,s:8x8x8x16x8,a:8x8x8x8x16,s:r23"; do
  gated ab-pow2.txt ./r29m.test -test.run '^TestIntelAB$' -intel.ab=7,40 -intel.set="$set"
done
for set in "131072=a:cur,s:cur,s:r23,s:8x8x8x16x16,s:8x8x8x8x8x4" "262144=a:cur,s:cur,s:r23,s:8x8x8x8x16x4" \
 "524288=a:cur,s:cur,s:r23,s:8x8x8x8x8x16" "1048576=a:cur,s:cur,s:r23,s:8x8x8x8x16x16"; do
  gated ab-pow2.txt ./r29m.test -test.run '^TestIntelAB$' -intel.ab=7,100 -intel.set="$set"
done
echo "# PHASE2 DONE $(date +%T)" >> ab-pow2.txt
