import subprocess, os, shutil, sys
root=os.environ['R22ROOT']  # the directory holding the fft clone and gocache
fft=root+'/fft'
gen=fft+'/internal/kernels/asmgen/amd64/gen.go'
orig=open(gen).read()
i0=orig.index('// float32 (complex64) Stockham pass kernels')
head,tail=orig[:i0],orig[i0:]
muts=[
 ('noblend','		e.raw("VBLENDPS $3, %s, %s, %s", e.v(y), e.v(dst), e.v(dst))\n',''),
 ('addsub-order','e.raw("VADDSUBPS %s, %s, %s", e.v(t2), e.v(t1), e.v(dst))','e.raw("VADDSUBPS %s, %s, %s", e.v(t1), e.v(t2), e.v(dst))'),
 ('dup-swapped','e.raw("VMOVSLDUP %s, %s", e.tw(j), e.v(t1))\n		e.raw("VMOVSHDUP %s, %s", e.tw(j), e.v(t2))','e.raw("VMOVSHDUP %s, %s", e.tw(j), e.v(t1))\n		e.raw("VMOVSLDUP %s, %s", e.tw(j), e.v(t2))'),
 ('r3-y2-sign','		e.add(1, 6, 7)\n		e.sub(2, 6, 7)\n		e.twStore(1, 1, 9','		e.add(1, 6, 7)\n		e.add(2, 6, 7)\n		e.twStore(1, 1, 9'),
 ('r5-c51c52','		e.mulk(9, 3, kC52)\n		e.mulk(10, 5, kC51)','		e.mulk(9, 3, kC51)\n		e.mulk(10, 5, kC51)'),
 ('r8-h-distributed','		e.add(8, 8, 11)\n		e.mulk(8, 8, kH) // a5','		e.mulk(8, 8, kH)\n		e.mulk(11, 11, kH)\n		e.add(8, 8, 11) // a5'),
 ('single-tw-halves','		e.raw("VMOVSHDUP X%d, X%d", t1, t2)\n		e.raw("VMOVSLDUP X%d, X%d", t1, t1)','		e.raw("VMOVSLDUP X%d, X%d", t1, t2)\n		e.raw("VMOVSHDUP X%d, X%d", t1, t1)'),
 ('last-gather','	e.raw("VMOVHPS %d(AX), X15, X15", 8*(3*r+j))','	e.raw("VMOVHPS %d(AX), X15, X15", 8*(2*r+j))'),
 ('pair-twnext','	x.body(r, false)\n	x.advance(16)\n	x.twNext(r)\n	b.Raw("JMP single")','	x.body(r, false)\n	x.advance(16)\n	b.Raw("JMP single")'),
 ('r4-rot-sign','		e.add(1, 5, 7)\n		e.sub(2, 4, 6)\n		e.sub(3, 5, 7)','		e.sub(1, 5, 7)\n		e.sub(2, 4, 6)\n		e.add(3, 5, 7)'),
]
env=dict(os.environ, GOWORK='off', GOTOOLCHAIN='local', GOCACHE=root+'/gocache', GOFLAGS='-mod=mod')
try:
  for name,a,b in muts:
    assert tail.count(a)==1,(name,tail.count(a))
    open(gen,'w').write(head+tail.replace(a,b))
    subprocess.check_call(['go','run','gen.go'],cwd=fft+'/internal/kernels/asmgen/amd64',env=env,stdout=subprocess.DEVNULL)
    shutil.move(fft+'/internal/kernels/asmgen/amd64/stockham32_amd64.s',fft+'/internal/kernels/stockham32_amd64.s')
    for f in ['cmul_amd64.s','butterfly_amd64.s','cpu_amd64.s']: os.remove(fft+'/internal/kernels/asmgen/amd64/'+f)
    e2=dict(env,GOOS='linux',GOARCH='amd64'); del e2['GOFLAGS']
    subprocess.check_call(['go','test','-c','-o',root+'/mut/'+name,'.'],cwd=fft,env=e2)
    print('built',name)
finally:
  open(gen,'w').write(orig)
  subprocess.check_call(['go','run','gen.go'],cwd=fft+'/internal/kernels/asmgen/amd64',env=env,stdout=subprocess.DEVNULL)
  for f in ['cmul_amd64.s','butterfly_amd64.s','cpu_amd64.s','stockham32_amd64.s']: shutil.move(fft+'/internal/kernels/asmgen/amd64/'+f, fft+'/internal/kernels/'+f)
