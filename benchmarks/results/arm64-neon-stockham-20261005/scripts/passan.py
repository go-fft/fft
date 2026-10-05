import sys, re, statistics as st
from collections import defaultdict
d = defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        m = re.match(r'BenchmarkSKPassNEON/(\S+?)/(go|neon|zip)(-\d+)?\s+\d+\s+([\d.]+) ns/op', l)
        if m: d[(m.group(1), m.group(2))].append(float(m.group(4)))
keys = []
for (k, v) in d:
    if k not in keys: keys.append(k)
print(f"{'pass':34} {'go ns':>9} {'neon ns':>9} {'go/neon':>8} {'spread g/n':>12} {'go/zip':>7} rounds")
for k in keys:
    g, n = d.get((k,'go')), d.get((k,'neon'))
    z = d.get((k,'zip'))
    if not g or not n: continue
    mg, mn = st.median(g), st.median(n)
    zs = f"{mg/st.median(z):7.3f}" if z else "      -"
    print(f"{k:34} {mg:9.0f} {mn:9.0f} {mg/mn:8.3f} {max(g)/min(g):5.2f}/{max(n)/min(n):5.2f} {zs} {len(g)}")
