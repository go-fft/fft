#!/usr/bin/env python3
"""f32an.py FILE: float32-releases.txt -> per row, medians of v0.12.0, v0.15.0,
v0.16.1 float32, the release ratios with their per-round ranges, and
f32 ÷ f64 of each release (each against its own float64)."""
import re,sys,statistics,collections
d=collections.defaultdict(lambda: collections.defaultdict(dict));cur=None;rnd=None;loads=[]
for l in open(sys.argv[1]):
    m=re.match(r'# round (\d+) variant (\S+) \S+ load (\S+)',l)
    if m: rnd=int(m.group(1));cur=m.group(2);loads.append(float(m.group(3)));continue
    m=re.match(r'Benchmark(\w+)/(\S+)\s+\d+\s+([\d.]+) ns/op',l)
    if m and cur: d[m.group(1)+'/'+m.group(2)][cur][rnd]=float(m.group(3))
print('# 1-min load at the start of each run',min(loads),'-',max(loads))
V=['f12.test','f15.test','f16.test']
med=lambda r,v: statistics.median(d[r][v].values())
rows=sorted(set(k.rsplit('/',1)[0] for k in d))
print(f"{'row':12s} {'kind':4s} {'f64 v16':>9s} {'f32 v12':>9s} {'f32 v15':>9s} {'f32 v16':>9s}  v12/v16 (range)  v15/v16 (range)  f32/f64 v12 v15 v16  spread")
for r in rows:
    for k in ['c','r','i']:
        a=r+'/'+k+'32'; b=r+'/'+k+'64'
        if a not in d: continue
        f=[med(a,v) for v in V]; g=[med(b,v) for v in V]
        sp=max(max(d[a][v].values())/min(d[a][v].values()) for v in d[a])
        rr=lambda x,y:[d[a][x][q]/d[a][y][q] for q in d[a][x] if q in d[a][y]]
        r12=rr(V[0],V[2]); r15=rr(V[1],V[2])
        print(f"{r:12s} {k:4s} {g[2]:9.0f} {f[0]:9.0f} {f[1]:9.0f} {f[2]:9.0f}  {f[0]/f[2]:5.2f} ({min(r12):.2f}-{max(r12):.2f}) {f[1]/f[2]:5.2f} ({min(r15):.2f}-{max(r15):.2f})  {f[0]/g[0]:.2f} {f[1]/g[1]:.2f} {f[2]/g[2]:.2f}  {sp:.2f}")
