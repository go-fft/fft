import sys,re,collections,statistics
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[m.group(1)].append(float(m.group(3)))
for k,v in d.items():
    print(f"{k:50s} {statistics.median(v):12.1f} spread {max(v)/min(v):.3f} n={len(v)}")
