import sys, math
rows={}
for l in open(sys.argv[1]):
    if not l.startswith('SWEEP'): continue
    p=l.split(); n=int(p[1]); c=[]
    for x in p[2:]:
        f,v=x.split('='); t,r=v.split('/'); c.append(([int(y) for y in f.split('x')],float(t),float(r)))
    rows[n]=c
def feats(n,f):
    d={}
    l1=1
    for k,r in enumerate(f):
        ido=n//(l1*r)
        if ido==1: key='F%d'%r
        else:
            key='T%d'%r
            if r in (4,8) and ido%4==0: key+='s'
            if r>=10 and (ido*16)%2048==0: key+='c'
        d[key]=d.get(key,0)+n
        l1*=r
    d['P']=d.get('P',0)+len(f)
    d['C']=1
    return d
keys=sorted({k for n,c in rows.items() for f,_,_ in c for k in feats(n,f)})
X=[];Y=[]
for n,c in rows.items():
    for f,t,_ in c:
        d=feats(n,f); X.append([d.get(k,0)/t for k in keys]); Y.append(1.0)
m=len(keys)
A=[[sum(x[i]*x[j] for x in X) for j in range(m)] for i in range(m)]
B=[sum(x[i]*y for x,y in zip(X,Y)) for i in range(m)]
for i in range(m): A[i][i]+=1e-9
# gaussian
for i in range(m):
    p=max(range(i,m),key=lambda r:abs(A[r][i])); A[i],A[p]=A[p],A[i]; B[i],B[p]=B[p],B[i]
    for r in range(m):
        if r!=i:
            fct=A[r][i]/A[i][i]
            for j in range(i,m): A[r][j]-=fct*A[i][j]
            B[r]-=fct*B[i]
coef={k:B[i]/A[i][i] for i,k in enumerate(keys)}
for k in keys: print('%-6s %.4f'%(k,coef[k]))
errs=[]; picks=[]
for n,c in sorted(rows.items()):
    preds=[sum(coef[k]*v for k,v in feats(n,f).items()) for f,_,_ in c]
    for (f,t,_),p in zip(c,preds): errs.append(abs(p-t)/t)
    i=min(range(len(c)),key=lambda i:preds[i])
    best=max(x[2] for x in c)
    picks.append((n,c[i][0],c[i][2],best))
print('mean rel err %.3f max %.3f'%(sum(errs)/len(errs),max(errs)))
g=math.exp(sum(math.log(b/r) for n,f,r,b in picks)/len(picks))
print('model pick: geo best/pick %.4f; worst vs cur %.3f'%(g,min(r for n,f,r,b in picks)))
for n,f,r,b in picks:
    if b/r>1.04 or r<1.0: print(n,f,'%.3f best %.3f'%(r,b))
