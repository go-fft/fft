#!/bin/sh
# Waits for the main parity run, keeps it, runs the branch parity run, then batch F.
cd ~/gofft-bench || exit 1
while pgrep -f run-remote.sh > /dev/null; do sleep 10; done
mkdir -p r21/parity-main && cp runpkg/run.log runpkg/benchmarks/*.txt runpkg/benchmarks/*.json runpkg/benchmarks/REPORT.md r21/parity-main/ 2>/dev/null
rm -rf runpkg && mkdir -p r21/y && tar xzf r21/runpkg-br.tgz -C r21/y && mv r21/y/runpkg-br runpkg
cd runpkg && GOV=go1.27.1 FFTW_DESC="3.3.10 built from source with --enable-neon" sh run-remote.sh arm64 "ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424 (perf-arm64-2)" > run.log 2>&1
cd ~/gofft-bench && mkdir -p r21/parity-br && cp runpkg/run.log runpkg/benchmarks/*.txt runpkg/benchmarks/*.json runpkg/benchmarks/REPORT.md r21/parity-br/ 2>/dev/null
cd r21 && tar xzf f.tgz && cd f && ./r21f.sh 15 > r21f.log 2>&1
echo CHAINDONE > ~/gofft-bench/r21/chain.done
