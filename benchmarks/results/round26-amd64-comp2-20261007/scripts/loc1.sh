#!/bin/sh
cd ~/r26
SETS='1000:rule,5x5x5x2x4,5x5x5x4x2;1080:rule,3x3x5x2x12,3x3x5x12x2,3x3x3x5x2x4;1296:rule,3x3x3x3x16,3x3x12x12,3x3x3x4x12;1920:rule,3x5x8x16,3x5x16x8,5x4x8x12,5x8x4x12;2000:rule,5x5x5x16,5x5x5x4x12x0'
SETS='1000:rule,5x5x5x2x4,5x5x5x4x2;1080:rule,3x3x5x2x12,3x3x5x12x2,3x3x3x5x2x4;1296:rule,3x3x3x3x16,3x3x12x12,3x3x3x4x12;1920:rule,3x5x8x16,3x5x16x8,5x4x8x12,5x8x4x12;2000:rule,5x5x5x16;6000:rule,3x5x5x5x16,5x5x5x4x12;256:rule,16x16,4x4x16;1024:rule,4x16x16,16x4x16;128:rule,16x8,4x4x8;512:rule,4x8x16,8x4x16;540:rule,3x3x5x12,3x3x3x5x4;500:rule,5x5x5x2x2;960:rule,3x5x4x16,3x5x16x4,5x4x4x12'
uptime
echo == passes
GOMAXPROCS=1 taskset -c 40 ./r26main.test -test.run '^TestR26Passes$' -r26.ab=7,100 -r26.sets="$SETS" -test.v 2>&1 | grep -E 'PASSES|FAIL|panic'
uptime
echo == real
GOMAXPROCS=1 taskset -c 40 ./r26main.test -test.run '^TestR26Real$' -r26.ab=9,100 -test.v 2>&1 | grep -E 'REAL|FAIL|panic'
uptime
echo == ab
GOMAXPROCS=1 taskset -c 40 ./r26main.test -test.run '^TestR26AB$' -r26.ab=9,100 -r26.sets="$SETS" -test.v 2>&1 | grep -E '^AB|FAIL|panic'
uptime
echo END
