#!/bin/zsh
cd ${0:a:h}/bin
while pgrep -f macab.sh > /dev/null; do sleep 20; done
waitload() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); if (( l < 3.0 )); then echo "load $(sysctl -n vm.loadavg)"; return; fi; sleep 20; done; }
for r in 1 2 3 4 5; do waitload; echo "== order round $r"; GOMAXPROCS=1 ./ord.arm64 -test.run '^$' -test.bench 'R22Order' -test.benchtime 0.2s; done > ../data/m4-order.txt 2>&1
echo DONE >> ../data/m4-order.txt
