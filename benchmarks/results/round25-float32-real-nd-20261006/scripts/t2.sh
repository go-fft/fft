#!/bin/sh
cd ~/r25
R='Strips32|BatchTwiddles32|Untangle32|Retangle32|RealPlan32Kernels'
{
echo "== kernels"; ./ktB2.amd64 -test.count 1 2>&1 | tail -3
echo "== ftB2 targeted"; ./ftB2.amd64 -test.count 1 -test.v -test.run "$R" 2>&1 | grep -E "^(--- |ok|FAIL|PASS)"
for i in 1 2 3 4 5 6 7; do echo "== batch mutant $i: $(sed -n ${i}p mutants-batch.txt)"; ./ftBM$i.amd64 -test.count 1 -test.run 'Strips32' 2>&1 | grep -E "^(--- FAIL|ok|PASS|FAIL)" | head -3; done
echo "== full suite ftB2"; ./ftB2.amd64 -test.count 1 2>&1 | tail -5
} > t2.log 2>&1
echo DONE >> t2.log
