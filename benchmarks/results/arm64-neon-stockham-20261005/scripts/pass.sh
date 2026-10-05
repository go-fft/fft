#!/bin/sh
# per-pass NEON vs Go, alternating inside each round (go then neon per pass)
cd ~/gofft-bench/neon || exit 1
B=${1:-fft-pass-v1.test}; R=${2:-5}; out=${3:-pass.txt}; : > $out
uptime >> $out
for r in $(seq 1 $R); do
  GOMAXPROCS=1 taskset -c 40 ./$B -test.run '^$' -test.bench 'SKPassNEON' -test.benchtime 200ms -test.count 1 | grep '^Benchmark' >> $out
done
uptime >> $out
echo DONE >> $out
