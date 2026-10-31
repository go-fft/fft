#!/bin/sh
# loc1.sh CORE: locate the 1024 time: FFTW plans, each pass of ours alone (default and FMA), FFTW again.
C=$1
cd ~/gofft-bench/z1024
echo "# start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
taskset -c $C ../plan1024 1024
GOMAXPROCS=1 taskset -c $C ./z43.amd64 -test.run 'TestZ43Passes$' -z43.ab=21,200 \
  -z43.sets='1024:rule,fmarule,32x32f,16x8x8,4x16x16,8x4x4x8,4x4x8x8'
GOMAXPROCS=1 taskset -c $C ./z43.amd64 -test.run 'TestZ43AB$' -z43.ab=21,200 \
  -z43.sets='1024:rule,fmarule,32x32f,8x16x8,16x8x8,4x16x16,16x16x4,8x4x4x8,4x4x8x8,4x8x4x8,8x8x4x4'
GOMAXPROCS=1 taskset -c $C ./z43.amd64 -test.run 'TestR26Overhead$' -r26.ab=21,200
taskset -c $C ../plan1024 1024
echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
echo END
