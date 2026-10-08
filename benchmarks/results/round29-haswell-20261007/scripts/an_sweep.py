import sys
rows={}
for l in open(sys.argv[1]):
    if not l.startswith('SWEEP'): continue
    p=l.split(); n=int(p[1]); c=[]
    for x in p[2:]:
        f,v=x.split('='); t,r=v.split('/'); c.append((f,float(t),float(r)))
    rows[n]=c
def cnt(n):
    e=a=b=0
    while n%2==0: n//=2; e+=1
    while n%3==0: n//=3; a+=1
    while n%5==0: n//=5; b+=1
    return e,a,b
import importlib
rule=None
if len(sys.argv)>2:
    exec(open(sys.argv[2]).read())
tot=[];loss=[]
for n,c in sorted(rows.items()):
    best=max(c,key=lambda x:x[2])
    line='%6d %s cur=%s best=%s %.3f'%(n,cnt(n),c[0][0],best[0],best[2])
    if rule:
        f=rule(n)
        if f is None: f=c[0][0]
        m=[x for x in c if x[0]==f]
        if m:
            r=m[0][2]; line+='  rule=%s %.3f (best/rule %.3f)'%(f,r,best[2]/r); tot.append(best[2]/r); loss.append(r)
        else: line+='  rule=%s NOT TIMED'%f
    print(line)
if tot:
    import math
    print('geo best/rule %.4f max %.3f ; rule vs cur min %.3f geo %.3f'%(math.exp(sum(map(math.log,tot))/len(tot)),max(tot),min(loss),math.exp(sum(map(math.log,loss))/len(loss))))
