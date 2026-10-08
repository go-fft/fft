import re,sys,math
rows={}
for l in open(sys.argv[1]):
    if l.startswith('HC '):
        p=l.split(); n=int(p[1]); v=[]
        for x in p[2:]:
            name,rest=x.split('=',1); t,r,rg=rest.split('/'); lo,hi=rg.split('-')
            v.append((name,float(t),float(r),float(lo),float(hi)))
        rows[n]=v
def g(xs): return math.exp(sum(map(math.log,xs))/len(xs)) if xs else float('nan')
cols={'cur/s':[], 'c2/i':[], 'c2/s':[]}
for n,v in sorted(rows.items()):
    cur_i=v[0]; cur_s=v[1]; c2=v[2:] or [ (cur_i[0],)+cur_i[1:], (cur_s[0],)+cur_s[1:] ]
    c2i,c2s=c2[0],c2[1]
    cols['cur/s'].append(cur_s[2]); cols['c2/i'].append(c2i[2]); cols['c2/s'].append(c2s[2])
    print('%6d %-14s %8.0f  cur/s %.3f [%.3f-%.3f]  c2 %-14s i %.3f [%.3f-%.3f] s %.3f [%.3f-%.3f]'%(n,cur_i[0][:-2],cur_i[1],cur_s[2],cur_s[3],cur_s[4],c2i[0][:-2],c2i[2],c2i[3],c2i[4],c2s[2],c2s[3],c2s[4]))
for k,x in cols.items(): print(k,'geo %.3f min %.3f max %.3f n %d'%(g(x),min(x),max(x),len(x)))
