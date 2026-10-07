#!/bin/sh
cd ~/r26
SETS='1000:rule,10x10x10,5x10x20,5x20x10,20x5x10,10x5x20,5x5x2x20,5x5x4x10,5x5x10x4,20x10x5;1080:rule,3x3x15x8,3x3x8x15,3x3x12x10,3x3x10x12,3x3x3x10x4,3x3x3x4x10,3x3x3x20x2,3x3x5x2x12,3x12x3x10;2000:rule,5x5x5x16,5x5x20x4,5x5x4x20,10x10x20,5x20x20,20x5x20,10x20x10;6000:rule,5x5x5x4x12,15x5x5x16,5x5x15x16,15x20x20,5x5x20x12,5x5x12x20,3x5x5x20x4,3x5x20x20;1920:rule,5x4x8x12,3x4x8x20,3x8x4x20,15x8x16,15x4x4x8,15x16x8;500:rule,5x5x20,5x20x5,5x10x10,10x5x10,20x5x5;540:rule,3x3x5x12,3x3x3x20,3x3x15x4,3x3x4x15,3x15x12,3x12x15;960:rule,5x4x4x12,15x8x8,3x4x4x20,15x4x16,3x8x4x10'
uptime
echo == ab
GOMAXPROCS=1 taskset -c 40 ./r26k.test -test.run '^TestR26AB$' -r26.ab=9,100 -r26.sets="$SETS" -test.v 2>&1 | grep -E '^AB|FAIL|panic'
uptime
echo == passes
GOMAXPROCS=1 taskset -c 40 ./r26k.test -test.run '^TestR26Passes$' -r26.ab=5,60 -r26.sets="$SETS" -test.v 2>&1 | grep -E 'PASSES|FAIL|panic'
uptime
echo END
