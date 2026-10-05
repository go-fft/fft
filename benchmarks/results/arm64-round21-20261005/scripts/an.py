#!/usr/bin/env python3
# Usage: an.py FILE [key-regex-to-group]  -> median ns and max/min spread per (tag,bench)
import re, sys, statistics as st
rows = {}
order = []
for line in open(sys.argv[1]):
    m = re.match(r'(.*?)\s*Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op', line)
    if not m: continue
    tags = ' '.join(t for t in m.group(1).split() if not t.startswith('r='))
    key = (tags, m.group(2))
    if key not in rows: order.append(key); rows[key] = []
    rows[key].append(float(m.group(4)))
for k in order:
    v = rows[k]
    print(f"{k[0]:12s} {k[1]:40s} n={len(v)} med={st.median(v):14.1f} spread={max(v)/min(v):.3f}")
