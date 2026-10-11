#!/bin/sh
cd ~/r37
while ps -eo pid,args | grep -q "[h]wq13b.sh"; do sleep 30; done
for s in 8x1024 32x1024 64x1024 128x1024 256x1024; do
  ./hwab.sh 10 1 rowsm-$s.txt 2 -test.run '^TestHwRows$' -hw.rows 11,8 -hw.shapes $s -hw.rowvars s:cur,s:4x8x4x8,s:4x4x4x16 -- br-amd64
done
./hwab.sh 10 1 e11-ab-15.txt 2 -test.run '^TestR26AB$' -r26.ab 15,100 -r26.sets '6144:rule,3x8x16x16,3x16x8x16;10240:rule,5x8x16x16,8x8x16x10;12288:rule,8x8x16x12' -- br-amd64
echo "# queue c done $(date +%T)" >> hwq13c.log
