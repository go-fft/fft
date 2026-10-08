#!/bin/sh
# parity30.sh ARCH CORE "MACHINE" V... : the parity harness for each variant, pinned, back to back.
A=$1; C=$2; M=$3; shift 3
cd ~/gofft-bench/r30
for v in "$@"; do
  echo "# $v start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c $C sh r30$v/runpkg/run-remote.sh $A "$M" > r30$v/parity-$(date +%H%M).log 2>&1
  cp -r r30$v/runpkg/benchmarks out-$v-$(date +%H%M)
  echo "# $v end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo END
