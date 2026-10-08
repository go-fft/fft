#!/bin/sh
cd ~/r29 && . ./gate.sh
for e in 6 7 8 9 10 11 12 13 14 15 16; do
  gated sweep-pow2.txt ./r29m.test -test.run '^TestIntelSweep$' -intel.sweep=3,20,$e,$e
done
for r in 6,1000 1001,4000 4001,9000 9001,16384; do
  gated comp.txt ./r29m.test -test.run '^TestHswComp$' -hsw.comp=5,40,${r}
done
for r in 6,1000 1001,4000 4001,9000 9001,16384; do
  lo=${r%,*}; hi=${r#*,}
  gated sweep-comp.txt ./r29m.test -test.run '^TestR26Sweep$' -r26.ab=5,40 -r26.min=$lo -r26.max=$hi
done
echo "# PHASE1 DONE $(date +%T)" >> comp.txt
