#!/bin/sh
# loc2.sh CORE: FFTW and go-fft with src at several offsets from dst.
C=$1
cd ~/gofft-bench/z1024
gcc -O2 -o fftwplace fftwplace.c -I../fftw/include ../fftw/lib/libfftw3.a -lm || exit 1
echo "# start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
taskset -c $C ./fftwplace 1024
GOMAXPROCS=1 taskset -c $C ./z43.amd64 -test.run 'TestZ43Place$' -z43.ab=21,100 -z43.sets='1024:rule,fmarule,8x4x4x8'
taskset -c $C ./fftwplace 1024
GOMAXPROCS=1 taskset -c $C ./z43.amd64 -test.run 'TestZ43Place$' -z43.ab=21,100 -z43.sets='1024:rule,fmarule,8x4x4x8'
echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
echo END
