#!/usr/bin/env python3
"""decan.py FILE: median ns/op (spread max/min) per benchmark, grouped by plan:
whole, sum of passes, and each pass."""
import re, sys, statistics, collections
rows = collections.OrderedDict()
for line in open(sys.argv[1]):
    m = re.match(r'Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op', line)
    if not m: continue
    rows.setdefault(m.group(1), []).append(float(m.group(3)))
plans = collections.OrderedDict()
for k, v in rows.items():
    if '0warm' in k: continue
    name = k.split('/', 1)[1]
    plan, part = name.rsplit('-', 1)
    plans.setdefault(plan, []).append((part, statistics.median(v), max(v)/min(v)))
for plan, parts in plans.items():
    whole = [p for p in parts if p[0] == 'whole'][0]
    ps = [p for p in parts if p[0] != 'whole']
    s = sum(p[1] for p in ps)
    print(f"{plan:24s} whole {whole[1]:7.0f} ({whole[2]:.2f})  sum {s:7.0f}  | " +
          "  ".join(f"{p[0]} {p[1]:.0f}" for p in ps))
