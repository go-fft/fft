#!/usr/bin/env python3
# aban.py FILE [A B]: per-row medians and A÷B time ratio per round (median, min-max) of an interleaved A/B file.
import re, sys, statistics as st
from collections import defaultdict
f = sys.argv[1]; A = sys.argv[2] if len(sys.argv) > 2 else "main"; B = sys.argv[3] if len(sys.argv) > 3 else "br"
t = defaultdict(lambda: defaultdict(dict)); rnd = var = None; order = []
for line in open(f):
    m = re.match(r"# round (\d+) variant (\S+)", line)
    if m:
        rnd, var = int(m.group(1)), m.group(2); continue
    m = re.match(r"Benchmark\w+/(\S+?)(?:-\d+)?\s+\d+\s+([\d.]+) ns/op", line)
    if m and var:
        row = m.group(1)
        if row not in order: order.append(row)
        t[row][var][rnd] = float(m.group(2))
print(f"{'row':8} {A+' ns':>10} {B+' ns':>10} {A+'/'+B:>7}  range         rounds")
for row in order:
    a, b = t[row][A], t[row][B]
    rs = [a[r] / b[r] for r in a if r in b]
    print(f"{row:8} {st.median(a.values()):10.1f} {st.median(b.values()):10.1f} {st.median(rs):7.3f}  {min(rs):.3f}-{max(rs):.3f}  {len(rs)}")
