#!/usr/bin/env python3
# gap grid: per n, median time of each gap relative to g=576 (576 time / gap time), spread in brackets
import re, sys, statistics as st
d = {}
for line in open(sys.argv[1]):
    m = re.match(r'r=(\d+) g=(\d+) BenchmarkGapSK/(\d+)\S*\s+\d+\s+([\d.]+) ns/op', line)
    if m: d.setdefault((int(m.group(3)), int(m.group(2))), []).append(float(m.group(4)))
ns = sorted({k[0] for k in d}); gs = sorted({k[1] for k in d})
base = int(sys.argv[2]) if len(sys.argv) > 2 else 576
print("time(g=%d)/time(g); spread max/min in brackets; n rounds=%d" % (base, len(d[(ns[0], gs[0])])))
print("%8s" % "n" + "".join("%17s" % ("g%d" % g) for g in gs))
for n in ns:
    b = st.median(d[(n, base)])
    print("%8d" % n + "".join("%10.3f [%.2f]" % (b / st.median(d[(n, g)]), max(d[(n, g)]) / min(d[(n, g)])) for g in gs))
