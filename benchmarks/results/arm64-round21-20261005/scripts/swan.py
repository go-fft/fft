import re,sys,statistics as st
d={}
for l in open(sys.argv[1]):
    m=re.match(r'r=\d+ BenchmarkSW/(\d+)/(\w+)\S*\s+\d+\s+([\d.]+) ns/op',l)
    if m: d.setdefault((int(m.group(1)),m.group(2)),[]).append(float(m.group(3)))
ns=sorted({k[0] for k in d}); ws=['w4','w8','w16','w32','w64']
print("lines time / strip time (median of %d; spread of the strip row)"%len(d[(ns[0],'lines')]))
for n in ns:
    b=st.median(d[(n,'lines')])
    print("%5d "%n+" ".join("%s %.2f[%.2f]"%(w,b/st.median(d[(n,w)]),max(d[(n,w)])/min(d[(n,w)])) for w in ws if (n,w) in d))
