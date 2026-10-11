#!/bin/sh
cd ~/r37
while ps -eo pid,args | grep -q "[h]wq13.sh"; do sleep 30; done
./hwab.sh 10 1 call-15.txt 2 -test.run '^TestHwUntangleCall$' -hw.call 15,20000 -- brk-amd64
echo "# queue b done $(date +%T)" >> hwq13b.log
