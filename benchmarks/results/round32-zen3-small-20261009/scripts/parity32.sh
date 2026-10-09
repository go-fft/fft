#!/bin/sh
# parity32.sh CORE V...: the parity harness for each variant (main, br), pinned, back to back,
# each in a fresh copy of its package; FFTW's best-of-9 small-size timer (fftwplan) after each.
C=$1; shift
cd ~/r32
i=0
for v in "$@"; do
  i=$((i+1)); d=par$i-$v
  rm -rf $d; mkdir $d; tar xzf parity-$v.tgz -C $d
  echo "# $d start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOV=go1.27.1 FFTW_DESC="3.3.10 built from source by setup.sh" taskset -c $C sh $d/runpkg/run-remote.sh amd64 "AMD EPYC 7773X (Zen 3), cfarm420" > $d/parity.log 2>&1
  echo "# $d exit $? end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  taskset -c $C ./fftwplan > $d/fftwplan.txt
done
echo END
