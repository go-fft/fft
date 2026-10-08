def r8max(e):
    f=[]; n8=e//3
    if e%3==1:
        if e>=4: n8-=1; f=[4,4]
        else: f=[2]
    elif e%3==2: f=[4]
    return [8]*n8+f
def rule(n):
    e=a=b=0; m=n
    while m%2==0: m//=2; e+=1
    while m%3==0: m//=3; a+=1
    while m%5==0: m//=5; b+=1
    if m!=1 or a+b==0 or e==0: return None
    if e>=10:
        if e==10: return 'x'.join(map(str,[3]*a+[5]*b+[8,8,16]))
        return None
    t12=min(a,e//2) if e<=9 else 0
    t20=min(b,(e-2*t12)//2) if e<=8 else 0
    ep=e-2*t12-2*t20
    t10=min(1,b-t20,ep) if (e<=7 and ep%2==1) else 0
    ep-=t10
    if ep==1 and t20>0: t20-=1; ep+=2
    elif ep==1 and t12>0 and e>=3: t12-=1; ep+=2
    t15=min(a-t12,b-t20-t10) if e<=6 else 0
    t15=min(t15,1)
    tail={4:[16],7:[8,16]}.get(ep) or list(reversed(r8max(ep)))
    f=[3]*(a-t12-t15)+[5]*(b-t20-t10-t15)+tail+[15]*t15+[10]*t10+[12]*t12+[20]*t20
    return 'x'.join(map(str,f))
