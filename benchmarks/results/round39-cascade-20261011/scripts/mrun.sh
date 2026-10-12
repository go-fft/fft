#!/bin/sh
# mrun.sh: run each mutant binary's whole suite on core 3; a mutant is caught when the suite fails.
cd ~/r39
: > mutants.txt
for m in M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14; do
  GOMAXPROCS=2 taskset -c 2,3 timeout 900 ./$m.test -test.count=1 > $m.log 2>&1; rc=$?
  if [ $rc = 0 ]; then echo "$m SURVIVED" >> mutants.txt; else echo "$m caught (exit $rc): $(grep -m2 -- '--- FAIL\|panic:' $m.log | tr '\n' ' ')" >> mutants.txt; fi
done
echo "# done $(date +%T)" >> mutants.txt
