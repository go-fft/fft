#!/usr/bin/env python3
"""med.py FILE: per variant and row, the median ns/op over rounds and max/min."""
import re,sys,statistics,collections
d=collections.defaultdict(lambda: collections.defaultdict(list));cur=None;loads=[]
for l in open(sys.argv[1]):
    m=re.match(r'# round (\d+) variant (\S+) \S+ load (\S+)',l)
    if m: cur=m.group(2); loads.append(float(m.group(3))); continue
    m=re.match(r'(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op',l)
    if m and cur: d[cur][m.group(1)].append(float(m.group(2)))
print(f"# load {min(loads):.2f}-{max(loads):.2f}")
for v,rows in d.items():
    for r,xs in rows.items():
        print(f"{v:10s} {r:45s} {statistics.median(xs):10.1f} {max(xs)/min(xs):5.2f} n={len(xs)}")
