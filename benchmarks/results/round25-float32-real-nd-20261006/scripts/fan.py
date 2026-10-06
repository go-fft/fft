import re,sys,statistics,collections
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'BenchmarkR25Fanout32/(\d+)/(\w+)\S*\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[(int(m.group(1)),m.group(2))].append(float(m.group(3)))
for n in sorted({k[0] for k in d}):
    a,b=d[(n,'serial')],d[(n,'rule')]
    rs=sorted(x/y for x,y in zip(a,b))
    print(n,len(a),"serial %.0f rule %.0f serial/rule %.2f range %.2f-%.2f IQR %.2f-%.2f"%(statistics.median(a),statistics.median(b),statistics.median(a)/statistics.median(b),rs[0],rs[-1],rs[len(rs)//4],rs[3*len(rs)//4]))
