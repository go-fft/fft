#!/bin/sh
# parity26.sh CORE "MACHINE" V... : the parity harness for each variant, pinned, back to back.
C=$1; M=$2; shift 2
cd ~/gofft-bench
for v in "$@"; do
  echo "# $v start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c $C sh r26$v/runpkg/run-remote.sh amd64 "$M" > r26$v/parity.log 2>&1
  cp -r r26$v/runpkg/benchmarks r26$v/out-$(date +%H%M)
  echo "# $v end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo END
