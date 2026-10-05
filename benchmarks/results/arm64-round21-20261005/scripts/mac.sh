#!/bin/sh
# Round 21, Apple M4 Max: gap x split grid and the strip widths, one goroutine; a round starts only when the 1-minute load is below 3.
cd "$(dirname "$0")" || exit 1
B=./exp.mac; R=${1:-7}; out=res/mac/gap-mac.txt; sw=res/mac/sw-mac.txt
: > $out; : > $sw
r=1
while [ $r -le $R ]; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if [ "$(echo "$l < 3" | bc)" != 1 ]; then sleep 20; continue; fi
  echo "round $r load $(sysctl -n vm.loadavg)" >> $out
  case $((r % 4)) in 0) G="576 1088 2112 4032";; 1) G="1088 2112 4032 576";; 2) G="2112 4032 576 1088";; 3) G="4032 576 1088 2112";; esac
  for g in $G; do for s in 1 0; do
    GAP=$g SPLIT=$s GOMAXPROCS=1 $B -test.run '^$' -test.bench 'GapSK' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r g=$g s=$s Benchmark/p" >> $out
  done; done
  echo "round $r end load $(sysctl -n vm.loadavg)" >> $out
  echo "round $r load $(sysctl -n vm.loadavg)" >> $sw
  GOMAXPROCS=1 $B -test.run '^$' -test.bench '^BenchmarkSW$' -test.benchtime 200ms -test.count 1 | sed -n "s/^Benchmark/r=$r Benchmark/p" >> $sw
  echo "round $r end load $(sysctl -n vm.loadavg)" >> $sw
  r=$((r+1))
done
echo DONE >> $out
