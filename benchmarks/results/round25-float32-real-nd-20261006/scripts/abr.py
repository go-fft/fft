import re,sys,statistics,collections,math
# A/B log: "== <kind> round R <bin>" headers; prints per-row medians, A/B ratio,
# pair range, and f32/f64 on each side.
A=sys.argv[2] if len(sys.argv)>2 else 'benchA'; B=sys.argv[3] if len(sys.argv)>3 else 'benchB'
d=collections.defaultdict(lambda: collections.defaultdict(list)); cur=None
for l in open(sys.argv[1]):
    m=re.match(r'== \S+ round (\d+) (\S+)',l)
    if m: cur=m.group(2).replace('ft','bench'); continue
    m=re.match(r'Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if m and cur: d[m.group(1)][cur].append(float(m.group(3)))
med=lambda x: statistics.median(x)
print("%-34s %10s %10s %6s %11s"%("bench","A ns","B ns","A/B","pair range"))
for k in sorted(d):
    a,b=d[k][A],d[k][B]
    if not a or not b: continue
    rs=[x/y for x,y in zip(a,b)]
    print("%-34s %10.0f %10.0f %6.2f %5.2f-%5.2f"%(k,med(a),med(b),med(a)/med(b),min(rs),max(rs)))
print("\nf32 time / f64 time (medians)")
def pair(k):
    for x,y in (('Real32_GoFFT','Real_GoFFT'),('CReal32_GoFFT','CReal_GoFFT'),('-f32','-f64')):
        if x in k: return k.replace(x,y)
for k in sorted(d):
    o=pair(k)
    if o and o in d and d[k][A] and d[k][B]:
        print("%-34s A %5.2f  B %5.2f"%(k,med(d[k][A])/med(d[o][A]),med(d[k][B])/med(d[o][B])))
