#!/bin/sh
# phase4: hardware counters, one variant per run, two event groups (4 programmable counters with HT on).
cd ~/r29 && . ./gate.sh
G1=cycles,instructions,uops_dispatched_port.port_0,uops_dispatched_port.port_1,uops_dispatched_port.port_5,resource_stalls.any
G2=cycles,l1d.replacement,ld_blocks_partial.address_alias,resource_stalls.sb,cycle_activity.stalls_l1d_pending
for v in "64=a:cur" "64=s:cur" "4096=a:cur" "4096=s:cur" "65536=a:cur" "65536=s:cur" "65536=s:8x8x8x8x16"; do
  for g in $G1 $G2; do
    echo "## $v" >> pmu.txt
    gated pmu.txt perf stat -x, -e $g ./r29b.test -test.run '^TestIntelAB$' -intel.ab=40,50 -intel.set="$v"
  done
done
for v in "3840:5x8x8x12" "3840:4x4x12x20" "3840:16x12x20" "1536:3x8x8x8" "1536:4x4x8x12" "1536:8x16x12"; do
  for g in $G1 $G2; do
    echo "## $v" >> pmu.txt
    gated pmu.txt perf stat -x, -e $g ./r29b.test -test.run '^TestR26AB$' -r26.ab=40,50 -r26.sets="$v"
  done
done
echo "# PHASE4 DONE $(date +%T)" >> pmu.txt
