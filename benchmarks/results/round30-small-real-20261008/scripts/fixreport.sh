#!/bin/sh
# fixreport.sh DIR...: report.py wants a -N suffix on the benchmark names, which a
# pinned run (GOMAXPROCS=1) does not print; add -1 and rebuild REPORT.md.
for d in "$@"; do
  ( cd $d && python3 - <<'PY'
import re
s=open("go_bench.txt").read()
s=re.sub(r"^(Benchmark\S+?)(\s+\d+\s+[\d.]+ ns/op)", lambda m: (m.group(1) if re.search(r"-\d+$", m.group(1)) else m.group(1)+"-1")+m.group(2), s, flags=re.M)
open("go_bench.txt","w").write(s)
PY
  FFTW_DESC="3.3.10 built from source by setup.sh" python3 report.py > /dev/null 2>&1 || echo "report failed in $d" )
done
