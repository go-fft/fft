import sys, re, statistics as st, math, collections
d = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(sys.argv[1]):
    m = re.match(r'go(\S+) Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op', line)
    if m: d[m.group(2)][m.group(1)].append(float(m.group(4)))
rows = []
for k, v in d.items():
    if len(v.get('1.26.4', [])) < 3 or len(v.get('1.27.1', [])) < 3: continue
    a, b = st.median(v['1.26.4']), st.median(v['1.27.1'])
    # spread: max/min within each toolchain, to judge noise
    sa = max(v['1.26.4']) / min(v['1.26.4']); sb = max(v['1.27.1']) / min(v['1.27.1'])
    rows.append((k, a, b, b / a, max(sa, sb), len(v['1.26.4'])))
for k, a, b, r, s, n in sorted(rows):
    flag = ' *' if abs(r - 1) > max(0.03, (s - 1)) else ''
    print(f'{k:34s} {a:14,.0f} {b:14,.0f} {r:6.3f}  spread {s:5.3f} n={n}{flag}')
g = math.exp(sum(math.log(r[3]) for r in rows) / len(rows))
print(f'geomean go1.27.1/go1.26.4 = {g:.4f} over {len(rows)} benchmarks; min {min(r[3] for r in rows):.3f} max {max(r[3] for r in rows):.3f}')
