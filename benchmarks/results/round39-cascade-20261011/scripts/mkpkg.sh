#!/bin/sh
# mkpkg.sh TREE OUT: the parity package for one commit (benchmarks/remote/README.md), with run-remote.sh
# patched to cd into its own directory.
set -e
TREE=$1; OUT=$2
rm -rf "$OUT"; mkdir -p "$OUT/benchmarks"
cd "$TREE/benchmarks"
GOWORK=off GOOS=linux GOARCH=amd64 go build -o "$OUT/verify.amd64" ./verify
GOWORK=off GOOS=linux GOARCH=amd64 go test -c -o "$OUT/bench.amd64" .
cp -r cbench fft_reference.py verify_correctness.py report.py "$OUT/benchmarks/"
sed 's|^cd ~/gofft-bench/runpkg/benchmarks$|cd "$(dirname "$0")/benchmarks"|' remote/run.sh > "$OUT/run-remote.sh"
grep -q 'dirname' "$OUT/run-remote.sh"
(cd "$OUT" && shasum -a 256 bench.amd64 verify.amd64 > SHA256SUMS)
