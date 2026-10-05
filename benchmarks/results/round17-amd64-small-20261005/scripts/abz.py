import sys,re,statistics as st,collections
# usage: abz.py file A B
f,A,B=sys.argv[1:4]
d=collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(f):
    m=re.match(r'(\S+) Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[m.group(2)][m.group(1)].append(float(m.group(4)))
for k,v in d.items():
    if A not in v or B not in v: continue
    a,b=st.median(v[A]),st.median(v[B])
    sp=max(max(v[A])/min(v[A]),max(v[B])/min(v[B]))
    r=a/b
    flag=' *' if abs(r-1)>max(0.02,sp-1) else ''
    print(f'{k:40s} {a:12.0f} {b:12.0f}  A/B {r:6.3f}  spread {sp:5.3f} n={len(v[A])}{flag}')
