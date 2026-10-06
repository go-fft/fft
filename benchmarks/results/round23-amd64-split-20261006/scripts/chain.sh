#!/bin/sh
cd ~/split23
./ab.sh 5 '^BenchmarkAB$' e2e-final-zen3.txt 40 300ms main.test br.test
./ab.sh 15 '^BenchmarkAB$/^(C|R|2D)$/^(32|64|128|256|1000|1009|1080|1296|1920|2000|6000|16384|65536)$' e2e15-final-zen3.txt 40 300ms main.test br.test
sh ~/gofft-bench/parity23.sh 40 > ~/split23/parity23.out 2>&1
echo CHAIN DONE >> ~/split23/parity23.out
