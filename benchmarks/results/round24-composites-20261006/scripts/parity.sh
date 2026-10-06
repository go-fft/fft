#!/bin/sh
# parity.sh ARCH CORE "MACHINE" : main then branch, pinned, back to back.
A=$1; C=$2; M=$3
cd ~/gofft-bench
for v in main br; do
  echo "# $v start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c $C sh r24$v/runpkg/run-remote.sh $A "$M" > r24$v/parity.log 2>&1
  echo "# $v end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
