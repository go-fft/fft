#!/bin/sh
cd ~/r25
R='Untangle32|Retangle32|RealPlan32Kernels'
{
echo "== kernels"; ./ktB.amd64 -test.count 1 -test.run 'Untangle32|Stockham' 2>&1 | tail -3; echo "exit $?"
echo "== ftB targeted"; ./ftB.amd64 -test.count 1 -test.v -test.run "$R" 2>&1 | grep -E "^(--- |ok|FAIL|PASS)"
for i in 1 2 3 4 5 6 7; do echo "== mutant $i: $(sed -n ${i}p mutants.txt)"; ./ftM$i.amd64 -test.count 1 -test.run "$R" 2>&1 | grep -E "^(--- FAIL|ok|PASS|FAIL)" | head -3; done
echo "== full suite ftB"; ./ftB.amd64 -test.count 1 2>&1 | tail -5
echo "== full kernels"; ./ktB.amd64 -test.count 1 2>&1 | tail -3
} > t1.log 2>&1
echo DONE >> t1.log
