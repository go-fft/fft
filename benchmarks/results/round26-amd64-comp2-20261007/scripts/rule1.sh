#!/bin/sh
cd ~/r26
uptime
GOMAXPROCS=1 taskset -c 40 ./r26b.test -test.run '^TestR26Rule$' -r26.ab=7,60 -test.v 2>&1 | grep -E '^RULE|FAIL|panic'
uptime
echo END
