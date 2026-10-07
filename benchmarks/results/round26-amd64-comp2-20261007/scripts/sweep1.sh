#!/bin/sh
cd ~/r26
uptime
GOMAXPROCS=1 taskset -c 40 ./r26s.test -test.run '^TestR26Sweep$' -r26.ab=5,40 -test.v 2>&1 | grep -E '^SWEEP|FAIL|panic'
uptime
echo END
