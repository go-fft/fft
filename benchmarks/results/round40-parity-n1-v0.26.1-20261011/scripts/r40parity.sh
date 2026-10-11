#!/bin/sh
# r40parity.sh RUN...: Round 40 parity of v0.26.1 on cfarm424, one pinned core
# (taskset -c 40), GOMAXPROCS=1; each run gated on a one-minute load below 1.5,
# the package's sha256 checked against r40-SHA256SUMS before it is unpacked.
cd ~/gofft-bench
for run in "$@"; do
  sha256sum -c --ignore-missing r40-SHA256SUMS | grep "r40pk.tgz: OK" || { echo "# $run sha256 MISMATCH"; exit 1; }
  rm -rf "r40$run"; mkdir "r40$run"; tar xzf r40pk.tgz -C "r40$run"
  while :; do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l<1.5)}" && break; echo "# $run waiting, load $l"; sleep 30; done
  echo "# $run start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOMAXPROCS=1 GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c 40 sh "r40$run/runpkg/run-remote.sh" arm64 "Neoverse-N1 (cfarm424), one pinned core" > "r40$run/parity.log" 2>&1
  echo "# $run exit $? end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "# done"
