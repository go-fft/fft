#!/bin/sh
cd ~/f32simd
for r in 1 2 3 4 5; do
  uptime
  GOMAXPROCS=1 taskset -c 40 ./bench0 -test.run '^$' -test.bench '^Benchmark(Complex|Real|CReal)(32)?_GoFFT$' -test.benchtime 0.3s
done
echo DONE
