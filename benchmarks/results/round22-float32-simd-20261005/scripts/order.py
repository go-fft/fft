import re,sys,statistics,collections
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'BenchmarkR22Order/(\d+)/(pf|odd)\S*\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[(int(m.group(1)),m.group(2))].append(float(m.group(3)))
for n in sorted({k[0] for k in d}):
    a,b=d[(n,'pf')],d[(n,'odd')]
    rs=[x/y for x,y in zip(a,b)]
    print("%8d pf %10.0f odd %10.0f  pf/odd %.3f  (%.2f-%.2f)"%(n,statistics.median(a),statistics.median(b),statistics.median(a)/statistics.median(b),min(rs),max(rs)))
