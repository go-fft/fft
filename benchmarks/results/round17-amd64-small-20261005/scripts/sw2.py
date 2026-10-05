import sys,re,statistics as st,collections
d=collections.OrderedDict()
for l in open(sys.argv[1]):
    m=re.match(r'Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m: d.setdefault(m.group(1),[]).append(float(m.group(3)))
g=collections.OrderedDict()
for k,v in d.items():
    kind,n,var=k.split('/')
    g.setdefault((kind,n),[]).append((var,st.median(v),max(v)/min(v)))
for (kind,n),rows in g.items():
    if kind=='StripWidth':
        base=rows[0][1]
        print(kind,n,' '.join(f'{v}:{base/m:.2f}' for v,m,s in rows), 'w0=%.0f'%base)
    else:
        cur=''.join(map(str,[]))
        best=min(rows,key=lambda r:r[1])
        first=rows[0]
        print(kind,n,'best',best[0],f'{best[1]:.0f}','| all:',' '.join(f'{v}:{m:.0f}' for v,m,s in sorted(rows,key=lambda r:r[1])[:6]))
