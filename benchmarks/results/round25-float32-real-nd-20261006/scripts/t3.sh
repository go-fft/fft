#!/bin/sh
cd ~/r25/b3
{
echo "== ftB2 targeted"; ./ftB2.amd64 -test.count 1 -test.v -test.run 'BatchPass32MatchesGo|Strips32' 2>&1 | grep -E "^(--- |ok|FAIL|PASS)"
for i in 2 3 5; do echo "== batch mutant $i: $(sed -n ${i}p mutants-batch.txt)"; ./ftBM$i.amd64 -test.count 1 -test.run 'Strips32|BatchPass32MatchesGo' 2>&1 | grep -E "^(--- FAIL|ok|PASS|FAIL)" | head -3; done
} > ../t3.log 2>&1
echo DONE >> ../t3.log
