import re,sys,statistics,collections
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'Benchmark(\w+?)(32)?_GoFFT/(\d+)\S*\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[(m.group(1),m.group(3),bool(m.group(2)))].append(float(m.group(4)))
print("%-8s %8s %10s %10s %6s  spread32 spread64"%("kind","n","f64 ns","f32 ns","f32/f64"))
for (k,n,is32),v in d.items():
    if is32: continue
    w=d[(k,n,True)]
    a,b=statistics.median(v),statistics.median(w)
    print("%-8s %8s %10.0f %10.0f %6.2f  %.2f %.2f"%(k,n,a,b,b/a,max(w)/min(w),max(v)/min(v)))
