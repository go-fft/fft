#!/bin/sh
# parity36.sh CORE V...: the parity harness for each variant (main, br), pinned, back to back,
# each in a fresh copy of its package (run-remote.sh patched to run in its own directory),
# its binaries checked against the SHA-256 recorded locally (V.sha) first.
C=$1; shift
cd ~/r36
i=0
for v in "$@"; do
  i=$((i+1)); d=par$i-$v
  mkdir $d || exit 1
  tar xzf parity-$v.tgz -C $d
  (cd $d/runpkg && sha256sum -c ../../$v.sha) || exit 1
  echo "# $d start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c $C sh $d/runpkg/run-remote.sh amd64 "AMD EPYC 7773X (Zen 3), cfarm420" > $d/parity.log 2>&1
  echo "# $d exit $? end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo END
