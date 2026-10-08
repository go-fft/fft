#!/bin/sh
# tests.sh: the full suites of the branch on this CPU, then the accuracy export.
cd ~/r29/e2e
echo "# tests $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
./br.test 2>&1 | tail -15
(cd ../e2e && ./kern.test 2>&1 | tail -5)
mkdir -p exp && ./br.test -test.run '^TestHswExport$' -hsw.export=exp -hsw.list=192,1000,1080,1536,1920,2000,3840,6000,6400,7680,15000,16000,512,8192,16384,524288,1048576
echo "# TESTS DONE $(date +%T)"
