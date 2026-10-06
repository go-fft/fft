import sys
S=sys.argv[1]; m=sys.argv[2]
g=open(S+'/mut/gen.go.orig').read(); k=open(S+'/mut/sk.go.orig').read()
def rep(s,a,b):
    assert a in s,(m,a); return s.replace(a,b,1)
if m=='blend': g=rep(g,'e.raw("VBLENDPD $1, %s, %s, %s", y(re)','e.raw("VBLENDPD $2, %s, %s, %s", y(re)')
if m=='rotinv': g=rep(g,'	if !e.inverse {\n		e.add(pr, ar, bi)','	if true {\n		e.add(pr, ar, bi)')
if m=='a7neg': g=rep(g,'		e.sub(6, 1, 0)\n		e.neg(0)\n','		e.sub(6, 1, 0)\n')
if m=='tworder': k=rep(k,'range [4]int{0, 2, 1, 3}','range [4]int{0, 1, 2, 3}')
if m=='store': g=rep(g,'	e.raw("VMOVUPD %s, %s", y(t0), a)\n	e.raw("VMOVUPD %s, %s", y(t1), at32(a))','	e.raw("VMOVUPD %s, %s", y(t1), a)\n	e.raw("VMOVUPD %s, %s", y(t0), at32(a))')
if m=='a5inv': g=rep(g,'		e.sub(3, 6, 9)\n		e.add(5, 9, 6)','		e.add(3, 6, 9)\n		e.add(5, 9, 6)')
if m=='load': g=rep(g,'	e.raw("VUNPCKLPD %s, %s, %s", y(t1), y(t0), y(re))','	e.raw("VUNPCKLPD %s, %s, %s", y(t0), y(t1), y(re))')
if m=='a7sum': g=rep(g,'		e.neg(0)\n		e.sub(0, 0, 1)','		e.add(0, 0, 1)\n		e.neg(0)')
if m=='noblend': g=rep(g,'	if first {\n		e.raw("VBLENDPD $1','	if false {\n		e.raw("VBLENDPD $1')
open(S+'/fft/internal/kernels/asmgen/amd64/gen.go','w').write(g); open(S+'/fft/internal/kernels/stockhamsplit_amd64.go','w').write(k)
