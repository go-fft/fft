import re,sys,statistics as st
d={}
for l in open(sys.argv[1]):
    m=re.match(r'r=\d+ BenchmarkSplit/(\d+)/(\w+)\S*\s+\d+\s+([\d.]+) ns/op',l)
    if m: d.setdefault((int(m.group(1)),m.group(2)),[]).append(float(m.group(3)))
ns=sorted({k[0] for k in d})
print("n  inter_med split_med  inter/split  spread(inter) spread(split)  per-round ratios min..max")
for n in ns:
    a,b=d[(n,'inter')],d[(n,'split')]
    rr=[x/y for x,y in zip(a,b)]
    print("%8d %12.0f %12.0f %6.3f  [%.2f] [%.2f]  %.3f..%.3f"%(n,st.median(a),st.median(b),st.median(a)/st.median(b),max(a)/min(a),max(b)/min(b),min(rr),max(rr)))
