#!/bin/sh
# parity.sh ARCH CORE "MACHINE" V1 V2 ... : each variant's parity run, pinned, back to back.
A=$1; C=$2; M=$3; shift 3
cd ~/gofft-bench
for v in "$@"; do
  d=${v%%:*}; tag=${v#*:}
  echo "# $tag start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c $C sh $d/runpkg/run-remote.sh $A "$M" > $d/parity-$tag.log 2>&1
  mkdir -p $d/out-$tag && cp $d/runpkg/benchmarks/REPORT.md $d/runpkg/benchmarks/*.json $d/runpkg/benchmarks/go_*.txt $d/out-$tag/
  echo "# $tag end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
