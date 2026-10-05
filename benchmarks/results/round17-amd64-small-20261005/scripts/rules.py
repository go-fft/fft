import sys,re,statistics as st,collections,math
cur={64:'88',128:'844',256:'884',512:'888',1024:'8844',2048:'8884',1000:'8555',1080:'83335',1296:'443333',1920:'84435',2000:'44555',6000:'443555'}
def rule(n,oddkey,powrev):
    f=cur[n]; odd=sorted([c for c in f if c in '357'],reverse=oddkey); p=[c for c in f if c in '248']
    if powrev: p=p[::-1]
    return ''.join(odd+p)
for fn in sys.argv[1:]:
    d=collections.defaultdict(list)
    for l in open(fn):
        m=re.match(r'BenchmarkOrders/(\d+)/(\d+)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
        if m: d[(int(m.group(1)),m.group(2))].append(float(m.group(4)))
    t={k:st.median(v) for k,v in d.items()}
    print(fn)
    rules={'current':lambda n:cur[n],'odd asc,pow':lambda n:rule(n,False,False),'odd desc,pow':lambda n:rule(n,True,False),'odd asc,pow rev':lambda n:rule(n,False,True),'odd desc,pow rev':lambda n:rule(n,True,True)}
    for name,fr in rules.items():
        rs=[]; out=[]
        for n in cur:
            best=min(v for (m,o),v in t.items() if m==n)
            o=fr(n); rs.append(t[(n,o)]/best); out.append(f'{n}:{t[(n,o)]/best:.3f}')
        g=math.exp(sum(map(math.log,rs))/len(rs))
        print(f'  {name:18s} geo {g:.3f} worst {max(rs):.3f} | '+' '.join(out))
