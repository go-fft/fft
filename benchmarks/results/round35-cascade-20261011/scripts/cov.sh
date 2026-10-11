#!/bin/sh
# cov.sh: build both packages' coverage test binaries for linux/amd64, run them on cfarm151, merge, report.
set -e
S=/private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent-r35
cd $S/fft
CP=$(go list ./... | paste -sd, -)
GOOS=linux GOARCH=amd64 go test -c -coverpkg=$CP -o ../covf.test . 2>/dev/null
GOOS=linux GOARCH=amd64 go test -c -coverpkg=$CP -o ../covk.test ./internal/kernels
cd ..
gzip -kf covf.test covk.test
./s151 covf.test.gz r35/ && ./s151 covk.test.gz r35/
./c151 'cd ~/r35 && gunzip -f covf.test.gz covk.test.gz && chmod +x cov*.test && sha256sum covf.test covk.test && (./covf.test -test.coverprofile=cf.out -test.timeout 40m > covf.log 2>&1; echo f=$?); (./covk.test -test.coverprofile=ck.out > covk.log 2>&1; echo k=$?); grep -A4 -- "--- FAIL" covf.log covk.log | head -40; tail -n 2 covf.log; tail -n 2 covk.log; (cat cf.out; tail -n +2 ck.out) > cov.out'
shasum -a 256 covf.test covk.test
./c151 'cat ~/r35/cov.out' > cov151.out
cd fft && go tool cover -func=../cov151.out | tail -1; go tool cover -func=../cov151.out | grep -v '100.0%' | head
