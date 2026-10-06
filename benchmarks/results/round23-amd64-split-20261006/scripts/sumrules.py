import sys,re,collections
d=collections.OrderedDict()
for l in open(sys.argv[1]):
    m=re.match(r'Benchmark\w+/(\d+)/(\S+)\s+\d+\s+([\d.]+) ns/op',l)
    if m:
        n,name,t=m.groups(); name=re.sub(r'-\d+$','',name); d.setdefault(n,[]).append((name,float(t)))
for n,v in d.items():
    cur=[t for nm,t in v if nm.startswith('cur')][0]
    s=sorted(v[1:],key=lambda x:x[1])
    print(n, 'cur=%s %.0f |'%([nm for nm,t in v if nm.startswith('cur')][0][4:],cur), ' '.join('%s:%.2f'%(nm[2:],cur/t) for nm,t in s[:4]))
