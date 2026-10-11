#!/bin/sh
# hwparity.sh RUN... (main1 br1 br2 main2): Round 37 parity on cfarm424, one pinned core
# (taskset -c 40), GOMAXPROCS=1; each run gated on a one-minute load below 1.5,
# the package's sha256 checked against SHA256SUMS before it is unpacked.
cd ~/gofft-bench
for run in "$@"; do
  v=${run%[0-9]}
  sha256sum -c --ignore-missing r37-SHA256SUMS | grep "r37pk-$v.tgz: OK" || { echo "# $run sha256 MISMATCH"; exit 1; }
  rm -rf "r37$run"; mkdir "r37$run"; tar xzf "r37pk-$v.tgz" -C "r37$run"
  while :; do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l<1.5)}" && break; echo "# $run waiting, load $l"; sleep 30; done
  echo "# $run start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOMAXPROCS=1 GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c 40 sh "r37$run/runpkg/run-remote.sh" arm64 "Neoverse-N1 (cfarm424), one pinned core" > "r37$run/parity.log" 2>&1
  echo "# $run exit $? end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "# done"
