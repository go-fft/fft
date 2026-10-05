import sys, re, statistics as st
from collections import defaultdict
d = defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r'g=(\d+) neon=(\d) BenchmarkGapSK/(\d+)(-\d+)?\s+\d+\s+([\d.]+) ns/op', l)
    if m: d[(int(m.group(3)), int(m.group(1)), int(m.group(2)))].append(float(m.group(5)))
ns = sorted({k[0] for k in d}); gs = sorted({k[1] for k in d})
cols = [(g, ne) for g in gs for ne in (0, 1)]
print("time of (gap 576, Go passes) / time of column; spread max/min in brackets")
print(f"{'n':>8} " + " ".join(f"{('g%d %s' % (g, 'neon' if ne else 'go')):>16}" for g, ne in cols))
for n in ns:
    base = st.median(d[(n, 576, 0)])
    row = []
    for c in cols:
        v = d[(n,)+c]
        row.append(f"{base/st.median(v):8.3f} [{max(v)/min(v):4.2f}]")
    print(f"{n:>8} " + " ".join(f"{x:>16}" for x in row))
