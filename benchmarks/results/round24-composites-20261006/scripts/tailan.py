import re,sys,statistics,collections
rows=collections.OrderedDict()
for l in open(sys.argv[1]):
    m=re.match(r'BenchmarkR24Tails/(\d+)-(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m: rows.setdefault((int(m.group(1)),m.group(2)),[]).append(float(m.group(4)))
by=collections.OrderedDict()
for (n,t),v in rows.items(): by.setdefault(n,[]).append((statistics.median(v),max(v)/min(v),t))
cur=dict(arg.split('=') for arg in sys.argv[2:])
for n,l in by.items():
    l.sort(); best=l[0]
    c=[x for x in l if x[2]==cur.get(str(n),'')]
    cs=f"current {c[0][2]} {c[0][0]/best[0]:.3f}" if c else ""
    print(f"{n:6d} best {best[2]:10s} {best[0]:8.0f} ({best[1]:.2f}) | 2nd {l[1][2]} {l[1][0]/best[0]:.3f} 3rd {l[2][2]} {l[2][0]/best[0]:.3f} | {cs}")
