import sys
# mutate.py FFTDIR MUTANT: applies one mutation to the Round 28 512-bit split generator
# (only in the section after its marker) or to its Go twiddle table.
D=sys.argv[1]; m=sys.argv[2]
gp=D+'/internal/kernels/asmgen/amd64/gen.go'; kp=D+'/internal/kernels/stockhamsplit512_amd64.go'
g=open(gp).read(); k=open(kp).read()
mark='type intelSplit512Emit struct'
head,tail=g.split(mark,1); tail=mark+tail
def rep(s,a,b):
    assert a in s,(m,a); return s.replace(a,b,1)
M={
 'blend':  ('g','Raw("MOVQ $1, R9").Raw("KMOVB R9, K1")','Raw("MOVQ $2, R9").Raw("KMOVB R9, K1")'),
 'noblend':('g','	if first {\n		// K1 = lane 0','	if false {\n		// K1 = lane 0'),
 'rotinv': ('g','	if !e.inverse {\n		e.add(pr, ar, bi)','	if true {\n		e.add(pr, ar, bi)'),
 'a7neg':  ('g','		e.sub(6, 1, 0)\n		e.neg(0)\n','		e.sub(6, 1, 0)\n'),
 'tworder':('k','range [8]int{0, 4, 1, 5, 2, 6, 3, 7}','range [8]int{0, 1, 2, 3, 4, 5, 6, 7}'),
 'store':  ('g','	e.raw("VMOVUPD %s, %s", intelZ(t0), a)\n	e.raw("VMOVUPD %s, %s", intelZ(t1), intelAt64(a))','	e.raw("VMOVUPD %s, %s", intelZ(t1), a)\n	e.raw("VMOVUPD %s, %s", intelZ(t0), intelAt64(a))'),
 'load':   ('g','	e.raw("VUNPCKLPD %s, %s, %s", intelZ(t1), intelZ(t0), intelZ(re))','	e.raw("VUNPCKLPD %s, %s, %s", intelZ(t0), intelZ(t1), intelZ(re))'),
 'a5inv':  ('g','		e.sub(3, 6, 9)\n		e.add(5, 9, 6)','		e.add(3, 6, 9)\n		e.add(5, 9, 6)'),
 'a7sum':  ('g','		e.neg(0)\n		e.sub(0, 0, 1)','		e.add(0, 0, 1)\n		e.neg(0)'),
 'spill':  ('g','e.raw("VADDPD 0(SP), %s, %s", intelZ(11), intelZ(7))','e.raw("VADDPD 128(SP), %s, %s", intelZ(11), intelZ(7))'),
 'twoff':  ('g','e.raw("VMOVUPD %d(R10), %s", off+64, intelZ(t[1]))','e.raw("VMOVUPD %d(R10), %s", off+32, intelZ(t[1]))'),
 'hconst': ('g','e.raw("VMULPD.BCST %d(R14), %s, %s", splitKH,','e.raw("VMULPD.BCST %d(R14), %s, %s", splitKNeg,'),
 'twnext': ('g','	e.body(false)\n	e.advance(128)\n	b.Raw("ADDQ $%d, R10", (r-1)*128)','	e.body(false)\n	e.advance(128)\n	b.Raw("ADDQ $%d, R10", (r-1)*64)'),
}
w,a,b=M[m]
if w=='g': tail=rep(tail,a,b)
else: k=rep(k,a,b)
open(gp,'w').write(head+tail); open(kp,'w').write(k)
