import re,sys,statistics,collections
# Medians (and min-max) of the branch-only R25 benchmarks across rounds.
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'BenchmarkR25(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m: d[m.group(1)].append(float(m.group(3)))
med={k:statistics.median(v) for k,v in d.items()}
print("Parts2D32 (ns, median of %d rounds): rows cols strips | f64 rows cols strips | cols/rows f32, strips/rows f32"%len(next(iter(d.values()))))
for n in [64,128,256,512,1024]:
    g=lambda s: med.get('Parts2D32/%d/%s'%(n,s),float('nan'))
    print("%5d  f32 %9.0f %9.0f %9.0f | f64 %9.0f %9.0f %9.0f | %.2f %.2f"%(n,g('f32-rows'),g('f32-cols'),g('f32-strips'),g('f64-rows'),g('f64-cols'),g('f64-strips'),g('f32-cols')/g('f32-rows'),g('f32-strips')/g('f32-rows')))
print("\nStripWidth32: time / best width per n")
for n in [64,128,256,512,1024]:
    ws={w:med['StripWidth32/%d/w%d'%(n,w)] for w in [4,8,16,32,64] if 'StripWidth32/%d/w%d'%(n,w) in med}
    if not ws: continue
    b=min(ws.values())
    print("%5d "%n+" ".join("w%d %.2f"%(w,t/b) for w,t in ws.items()))
print("\nUntangle32 (Go time / kernel time)")
for n in [256,1024,4096,65536,1000,1920]:
    for k in ['untangle','retangle']:
        a,b=med.get('Untangle32/%d/%s-go'%(n,k)),med.get('Untangle32/%d/%s-simd'%(n,k))
        if a and b: print("%6d %s %.2f (%.0f / %.0f ns)"%(n,k,a/b,a,b))
