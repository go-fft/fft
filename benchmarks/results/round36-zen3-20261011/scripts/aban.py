#!/usr/bin/env python3
"""aban.py FILE A B: per row, the median over rounds of A's row median ÷ B's (and range), from abn.sh output of TestZmPlace."""
import re, statistics, sys
f, A, B = sys.argv[1:4]
cur = None; rnd = None; data = {}
for line in open(f):
    m = re.match(r"# round (\d+) variant (\S+)", line)
    if m:
        rnd, cur = int(m.group(1)), m.group(2); continue
    m = re.match(r"ROW (\S+)\s+median\s+([\d.]+)", line)
    if m and cur:
        data.setdefault(m.group(1), {}).setdefault(rnd, {})[cur] = float(m.group(2))
for row, rs in data.items():
    rat = [v[A] / v[B] for v in rs.values() if A in v and B in v]
    a = statistics.median(v[A] for v in rs.values() if A in v); b = statistics.median(v[B] for v in rs.values() if B in v)
    print(f"{row:8} {A} {a:11.1f} {B} {b:11.1f}  {A}/{B} {statistics.median(rat):.3f}  range {min(rat):.3f}-{max(rat):.3f}  ({len(rat)} rounds)")
