#!/usr/bin/env python3
"""aban.py FILE A B: per row, median ns of A and B, A/B ratio of medians,
range of per-round ratios, and the larger of the two variants' max/min spread."""
import re,sys,statistics,collections
f,A,B=sys.argv[1:4]
d=collections.defaultdict(lambda: collections.defaultdict(dict))
cur=None;rnd=None
for l in open(f):
    m=re.match(r'# round (\d+) variant (\S+)',l)
    if m: rnd=int(m.group(1)); cur=m.group(2); continue
    m=re.match(r'Benchmark\S+?/(\S+)\s+\d+\s+([\d.]+) ns/op',l)
    if m and cur and m.group(1)!='0warm': d[m.group(1)][cur][rnd]=float(m.group(2))
print(f"{'row':10s} {A:>10s} {B:>10s}  {A}/{B}  per-round range   spread")
for row,v in d.items():
    a,b=v[A],v[B]
    rs=[a[r]/b[r] for r in a if r in b]
    ma,mb=statistics.median(a.values()),statistics.median(b.values())
    sp=max(max(a.values())/min(a.values()),max(b.values())/min(b.values()))
    flag='' if (min(rs)>1 or max(rs)<1) else '  (inside)'
    print(f"{row:10s} {ma:10.0f} {mb:10.0f}  {ma/mb:6.3f}  {min(rs):.3f}-{max(rs):.3f}  {sp:.2f}{flag}")
