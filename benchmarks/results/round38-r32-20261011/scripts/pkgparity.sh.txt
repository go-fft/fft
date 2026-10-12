#!/bin/sh
# pkgparity.sh SRC NAME: cross-compile the parity package of the tree SRC into $S/par/NAME/runpkg, tar it to $S/par/parity-NAME.tgz
set -e
S=${S:?set S to a work directory outside the tree}
SRC=$1; NAME=$2
O=$S/par/$NAME/runpkg
mkdir -p $O/benchmarks
cd $SRC/benchmarks
GOWORK=off GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o $O/verify.amd64 ./verify
GOWORK=off GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -c -o $O/bench.amd64 .
cp -r cbench fft_reference.py verify_correctness.py report.py $O/benchmarks/
# The stock script changes to a fixed ~/gofft-bench/runpkg; run in its own directory instead.
sed 's|^cd ~/gofft-bench/runpkg/benchmarks$|cd "$(dirname "$0")/benchmarks"|' remote/run.sh > $O/run-remote.sh
grep -q 'dirname' $O/run-remote.sh
(cd $O && sha256sum verify.amd64 bench.amd64 2>/dev/null || shasum -a 256 verify.amd64 bench.amd64) > $S/par/$NAME.sha
cd $S/par/$NAME && tar czf ../parity-$NAME.tgz runpkg && ls -l ../parity-$NAME.tgz && cat ../$NAME.sha
