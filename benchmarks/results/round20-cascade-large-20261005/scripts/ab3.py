import sys,re,statistics as st,collections
f=sys.argv[1]; bins=sys.argv[2:]
d=collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(f):
    m=re.match(r'(\S+) Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[m.group(2)][m.group(1)].append(float(m.group(4)))
base=bins[0]
print(f'{"row":28s} {base:>12s} '+' '.join(f'{b+"÷":>18s}' for b in bins[1:]))
for k,v in d.items():
    a=st.median(v[base]); sa=max(v[base])/min(v[base])
    cells=[]
    for b in bins[1:]:
        m=st.median(v[b]); s=max(sa,max(v[b])/min(v[b]))
        cells.append(f'{a/m:6.3f} (sp {s:4.2f})')
    print(f'{k:28s} {a:12.0f} '+' '.join(f'{c:>18s}' for c in cells))
