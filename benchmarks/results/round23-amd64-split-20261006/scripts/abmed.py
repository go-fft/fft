import sys,re,collections,statistics
# ab file: "=== round i bin X load ..." then bench lines. Prints per-bench median per bin and ratio A/B.
d=collections.defaultdict(lambda: collections.defaultdict(list))
binname=None; bins=[]
for l in open(sys.argv[1]):
    m=re.match(r'=== round \d+ bin (\S+)',l)
    if m:
        binname=m.group(1)
        if binname not in bins: bins.append(binname)
        continue
    m=re.match(r'Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m and binname: d[m.group(1)][binname].append(float(m.group(3)))
A,B=bins[0],bins[1]
print(f"{'bench':55s} {A:>14s} {B:>14s}  A/B  spreadA spreadB")
for k,v in d.items():
    a,b=v[A],v[B]
    if not a or not b: continue
    ma,mb=statistics.median(a),statistics.median(b)
    print(f"{k:55s} {ma:14.1f} {mb:14.1f} {ma/mb:5.3f} {max(a)/min(a):6.3f} {max(b)/min(b):6.3f}")
