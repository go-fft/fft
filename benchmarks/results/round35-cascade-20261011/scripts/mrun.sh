#!/bin/sh
# mrun.sh PKG TESTREGEX: regenerate radix9_amd64.s and zntwopass_amd64.s from gen.go, build PKG's
# linux/amd64 test binary, run TESTREGEX on cfarm151; exit with the test's status.
S=/private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent-r35
cd $S/fft/internal/kernels/asmgen/amd64 || exit 3
GOFLAGS=-mod=mod go run gen.go > /dev/null || exit 3
cp radix9_amd64.s zntwopass_amd64.s ../../
rm -f cmul_amd64.s butterfly_amd64.s cpu_amd64.s stockham32_amd64.s stockhamsplit_amd64.s untangle32_amd64.s batch32_amd64.s comp2_amd64.s stockhamsplit512_amd64.s untangle_amd64.s radix9_amd64.s zntwopass_amd64.s
cd $S/fft && GOOS=linux GOARCH=amd64 go test -c -o $S/mut.test "$1" || exit 3
cd $S && gzip -f mut.test && ./s151 mut.test.gz r35/ && ./c151 "cd ~/r35 && gunzip -f mut.test.gz && chmod +x mut.test && ./mut.test -test.run '$2' 2>&1 > m.log 2>&1; s=$?; tail -n 4 m.log; exit $s"
