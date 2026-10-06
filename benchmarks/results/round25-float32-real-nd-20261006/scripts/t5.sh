#!/bin/sh
cd ~/r25/b5
{
echo "== ftB5 targeted"; ./ftB5.amd64 -test.count 1 -test.v -test.run 'BatchPass32MatchesGo|Strips32|BatchTwiddles32' 2>&1 | grep -E "^(--- |ok|FAIL|PASS)"
for i in 1 2 3 4 5 6 7; do echo "== batch mutant $i: $(sed -n ${i}p mutants-batch.txt)"; ./ftBM$i.amd64 -test.count 1 -test.run 'Strips32|BatchPass32MatchesGo' 2>&1 | grep -E "^(--- FAIL|ok|PASS|FAIL)" | head -3; done
echo "== full suite ftB5"; ./ftB5.amd64 -test.count 1 2>&1 | tail -3
} > ../t5.log 2>&1
echo DONE >> ../t5.log
