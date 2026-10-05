#!/usr/bin/env python3
# A/B: main time / branch time per row (median of each), max/min spread of each side, rounds.
import re,sys,statistics as st
d={}; order=[]
for l in open(sys.argv[1]):
    m=re.match(r'r=(\d+) v=(\w+) Benchmark(\S+?)(-\d+)?\s+\d+\s+([\d.]+) ns/op',l)
    if not m: continue
    k=m.group(3)
    if k not in order: order.append(k)
    d.setdefault((k,m.group(2)),[]).append(float(m.group(5)))
print("%-16s %6s %12s %12s %7s %7s %s"%("row","rounds","main ns","branch ns","ratio","spreadM","spreadB"))
for k in order:
    a,b=d.get((k,'main'),[]),d.get((k,'br'),[])
    if not a or not b: continue
    print("%-16s %6d %12.0f %12.0f %7.3f %7.2f %7.2f"%(k,len(a),st.median(a),st.median(b),st.median(a)/st.median(b),max(a)/min(a),max(b)/min(b)))
