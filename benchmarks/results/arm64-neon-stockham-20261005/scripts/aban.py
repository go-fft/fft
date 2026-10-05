# Interleaved A/B: lines "<label> Benchmark<name> N <ns> ns/op"; prints main time / new time
# (above 1 = the branch is faster), the spread (max/min within each side), and
# flags rows whose change is outside max(3%, spread).
import sys, re, statistics as st, math, collections
A, B = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('main', 'neon')
d = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(sys.argv[1]):
    m = re.match(r'(\S+) Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op', line)
    if m: d[m.group(2)][m.group(1)].append(float(m.group(4)))
rows = []
for k, v in d.items():
    if len(v.get(A, [])) < 3 or len(v.get(B, [])) < 3: continue
    a, b = st.median(v[A]), st.median(v[B])
    sa = max(v[A]) / min(v[A]); sb = max(v[B]) / min(v[B])
    rows.append((k, a, b, a / b, max(sa, sb), len(v[A])))
print(f'{"row":34s} {A+" ns":>14} {B+" ns":>14} {A+"/"+B:>9}  spread')
for k, a, b, r, s, n in sorted(rows):
    flag = ' *' if abs(r - 1) > max(0.03, (s - 1)) else ''
    print(f'{k:34s} {a:14,.0f} {b:14,.0f} {r:9.3f}  {s:5.3f} n={n}{flag}')
g = math.exp(sum(math.log(r[3]) for r in rows) / len(rows))
print(f'geomean {A}/{B} = {g:.4f} over {len(rows)} rows; min {min(r[3] for r in rows):.3f} max {max(r[3] for r in rows):.3f}')
