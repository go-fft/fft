import re,sys,statistics,collections
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'Benchmark\w+/(\d+)/r=(\d+)/ido=(\d+)/l1=(\d+)/(go|simd|neon)\S*\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[m.groups()[:4]+(m.group(5),)].append(float(m.group(6)))
keys=sorted({k[:4] for k in d},key=lambda k:(int(k[0]),-int(k[2])))
print("%8s %2s %7s %7s %10s %10s %6s %6s"%("n","r","ido","l1","go ns","simd ns","ratio","range"))
for k in keys:
    g=d.get(k+('go',)); s=d.get(k+('simd',)) or d.get(k+('neon',))
    if not g or not s: continue
    rs=[a/b for a,b in zip(g,s)]
    print("%8s %2s %7s %7s %10.0f %10.0f %6.2f %4.2f-%4.2f"%(k+(statistics.median(g),statistics.median(s),statistics.median(g)/statistics.median(s),min(rs),max(rs))))
