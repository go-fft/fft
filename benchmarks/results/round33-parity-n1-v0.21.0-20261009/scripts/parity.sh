#!/bin/sh
# r33parity.sh RUN... : v0.21.0 parity, one pinned core (taskset -c 40), GOMAXPROCS=1,
# each run gated on a one-minute load below 1.5, load logged at start and end.
cd ~/gofft-bench
for v in "$@"; do
  rm -rf r33$v; mkdir r33$v; tar xzf r33pk.tgz -C r33$v
  while :; do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l<1.5)}" && break; echo "# $v waiting, load $l"; sleep 30; done
  echo "# $v start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOMAXPROCS=1 GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c 40 sh r33$v/runpkg/run-remote.sh arm64 "Neoverse-N1 (cfarm424), one pinned core" > r33$v/parity.log 2>&1
  echo "# $v exit $? end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "# done"
