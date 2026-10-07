import re,collections,sys
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.match(r'SW (\d+) (\S+)\s+([\d.]+) ns',l)
    if m: d[int(m.group(1))].append((m.group(2),float(m.group(3))))
for n,rows in d.items():
    cur=[t for nm,t in rows if nm=='w:cur'][0]
    print(n,'w:cur %.0f'%cur, ' | '.join(f"{nm} {cur/t:.3f}" for nm,t in rows[:8]))
