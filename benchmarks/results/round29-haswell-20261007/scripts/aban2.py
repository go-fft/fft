"""aban2.py FILE A B: per row, the ratio of medians A/B, the 2nd-lowest and
2nd-highest per-round ratio (one outlier round dropped at each end), and how
many rounds were above 1."""
import re,sys,statistics,collections
f,A,B=sys.argv[1:4]
d=collections.defaultdict(lambda: collections.defaultdict(dict));cur=rnd=None
for l in open(f):
    m=re.match(r'# round (\d+) variant (\S+)',l)
    if m: rnd=int(m.group(1)); cur=m.group(2); continue
    m=re.match(r'Benchmark\S+?/(\S+)\s+\d+\s+([\d.]+) ns/op',l)
    if m and cur: d[m.group(1)][cur][rnd]=float(m.group(2))
for row,v in d.items():
    a,b=v[A],v[B]; rs=sorted(a[r]/b[r] for r in a if r in b)
    print(f"{row:10s} {statistics.median(a.values())/statistics.median(b.values()):.3f}  {rs[1]:.3f}-{rs[-2]:.3f}  above1 {sum(x>1 for x in rs)}/{len(rs)}")
