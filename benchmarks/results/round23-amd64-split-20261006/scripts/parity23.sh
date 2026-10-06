#!/bin/sh
# parity23.sh CPU: the parity harness for main then the branch, one pinned core.
cd ~/gofft-bench
for v in main23 split23; do
  echo "=== $v load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOV=go1.27.1 FFTW_DESC="3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma" \
    taskset -c "$1" sh runpkg-$v/run-remote.sh amd64 "AMD EPYC 7773X (Zen 3), cfarm420, one pinned core" > parity-$v.log 2>&1
  echo "=== $v done load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo DONE
