#!/bin/sh
# loc3.sh CORE: the FMA-pipe prototype kernel sets against the default ones, in one process, twice.
C=$1
cd ~/gofft-bench/z1024
echo "# start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
GOMAXPROCS=1 taskset -c $C ./z43b.amd64 -test.run 'TestZ43Bal$' -z43.ab=21,100
GOMAXPROCS=1 taskset -c $C ./z43b.amd64 -test.run 'TestZ43Bal$' -z43.ab=21,100
echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
echo END
