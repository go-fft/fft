import re,sys,statistics,collections,math
# e2e log with "== e2e round R binary" headers; A=bench0 B=bench1 (or given)
A=sys.argv[2] if len(sys.argv)>2 else 'bench0'; B=sys.argv[3] if len(sys.argv)>3 else 'bench1'
d=collections.defaultdict(lambda: collections.defaultdict(list)); cur=None
for l in open(sys.argv[1]):
    m=re.match(r'== \S+ round (\d+) (\S+)',l)
    if m: cur=m.group(2); continue
    m=re.match(r'Benchmark(\S+)\s+\d+\s+([\d.]+) ns/op',l)
    if m and cur: d[m.group(1)][cur].append(float(m.group(2)))
print("%-28s %10s %10s %6s %11s"%("bench","A ns","B ns","A/B","pair range"))
gm=collections.defaultdict(list)
for k,v in d.items():
    a,b=v[A],v[B]
    if not a or not b: continue
    rs=[x/y for x,y in zip(a,b)]
    r=statistics.median(a)/statistics.median(b)
    gm['32' in k].append(r)
    print("%-28s %10.0f %10.0f %6.2f %5.2f-%5.2f"%(k,statistics.median(a),statistics.median(b),r,min(rs),max(rs)))
for k,v in gm.items(): print('geomean', 'f32' if k else 'f64', round(math.exp(sum(map(math.log,v))/len(v)),3), len(v))
