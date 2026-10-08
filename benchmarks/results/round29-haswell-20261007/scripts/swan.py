import re,sys,collections
rows={};fresh=set()
for l in open(sys.argv[1]):
    if l.startswith('# start'): fresh=set()
    m=re.match(r'SW (\d+) (\S+)\s+([\d.]+) ns\s+first/this ([\d.]+)\s+spread ([\d.]+)',l)
    if m:
        n=int(m[1])
        if n not in fresh: rows[n]=[]; fresh.add(n)   # keep the last try only
        rows[n].append((m[2],float(m[3]),float(m[4]),float(m[5])))
for n,r in sorted(rows.items()):
    d={x[0]:x for x in r}
    cur=d['a:cur']
    ba=[x for x in r if x[0].startswith('w:')][:3]
    bs=[x for x in r if x[0].startswith('s:')][:4]
    print(n,'cur %.0f'%cur[1],'| i:',' '.join('%s %.3f'%(x[0][2:],x[2]) for x in ba),'| s:',' '.join('%s %.3f'%(x[0][2:],x[2]) for x in bs))
