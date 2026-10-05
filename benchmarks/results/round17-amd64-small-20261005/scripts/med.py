import sys,re,statistics as st,collections
d=collections.OrderedDict()
for f in sys.argv[1:]:
  for l in open(f):
    m=re.match(r'Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m: d.setdefault(m.group(1),[]).append(float(m.group(3)))
for k,v in d.items(): print(f'{k:40s} {st.median(v):12.1f}  spread {max(v)/min(v):.3f}')
