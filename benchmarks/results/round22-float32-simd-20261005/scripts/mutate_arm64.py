import subprocess, os, shutil
root=os.environ['R22ROOT']  # the directory holding the fft clone and gocache
fft=root+'/fft'
gdir=fft+'/internal/kernels/asmgen/arm64'
gen=gdir+'/gen.go'
orig=open(gen).read()
i0=orig.index('// float32 (complex64) Stockham pass kernels')
head,tail=orig[:i0],orig[i0:]
muts=[
 ('imag-other-product','			b.VFMUL4S(31, yi, 28).VFMLA4S(31, yr, 29) // im','			b.VFMUL4S(31, yr, 29).VFMLA4S(31, yi, 28) // im'),
 ('pass2-imag','			b.VFMUL4S(31, yr, 29).VFMLA4S(31, yi, 28) // im = yr','			b.VFMUL4S(31, yi, 28).VFMLA4S(31, yr, 29) // im = yr'),
 ('no-restore','			b.Raw("VMOV V%d.S[0], V30.S[0]", yr).Raw("VMOV V%d.S[0], V31.S[0]", yi)','			_ = yi'),
 ('r8-hP-unfused','	b.VFMLA4S(14, 2, 18).VFMLA4S(15, 3, 18)','	b.VFMUL4S(30, 2, 18).VFADD4S(14, 14, 30).VFMUL4S(31, 3, 18).VFADD4S(15, 15, 31)'),
 ('r8-R-negsum','		b.VFNEG4S(6, 24)\n		sub(6, 6, 25)  // R','		add(6, 24, 25)\n		b.VFNEG4S(6, 6)  // R'),
 ('r3-half-unfused','	b.VFMLS4S(0, 6, 24).VFMLS4S(1, 7, 24) // ca','	b.VFMUL4S(28, 6, 24).VFSUB4S(0, 0, 28).VFMUL4S(29, 7, 24).VFSUB4S(1, 1, 29) // ca'),
 ('r3-y2-sign','	aPlusSz(10, 0, 3) // y2 = ca - rotS(q)','	aMinusSz(10, 0, 3) // y2 = ca - rotS(q)'),
 ('r5-u-other','		b.VFMUL4S(20+k, 4+k, 27).VFMLA4S(20+k, 2+k, 26)','		b.VFMUL4S(20+k, 2+k, 26).VFMLA4S(20+k, 4+k, 27)'),
 ('r5-c-swap','		b.VFMUL4S(18+k, 12+k, 24).VFMLA4S(18+k, 10+k, 25)','		b.VFMUL4S(18+k, 12+k, 25).VFMLA4S(18+k, 10+k, 25)'),
 ('lone-store-swap','		b.Raw("FSTPS (F%d, F%d), (%s)", v, v+1, ptr)','		b.Raw("FSTPS (F%d, F%d), (%s)", v+1, v, ptr)'),
 ('last5-ext','			b.Raw("VEXT $8, V%d.B16, V%d.B16, V%d.B16", t[3], t[0], e1)','			b.Raw("VEXT $8, V%d.B16, V%d.B16, V%d.B16", t[0], t[3], e1)'),
 ('last8-zip','				Raw("VZIP1 V%d.D2, V%d.D2, V%d.D2", 28+i, 24+i, 16+i).\n				Raw("VZIP2 V%d.D2, V%d.D2, V%d.D2", 28+i, 24+i, 20+i)','				Raw("VZIP2 V%d.D2, V%d.D2, V%d.D2", 28+i, 24+i, 16+i).\n				Raw("VZIP1 V%d.D2, V%d.D2, V%d.D2", 28+i, 24+i, 20+i)'),
 ('pair-tw-stride','		b.Raw("VLD1.P 16(R14), [V28.S2, V29.S2]")','		b.Raw("VLD1.P 32(R14), [V28.S2, V29.S2]")'),
 ('r8-T-fwd','		b.VFNEG4S(7, 25)\n		sub(7, 7, 24) // T','		add(7, 25, 24)\n		b.VFNEG4S(7, 7) // T'),
]
env=dict(os.environ, GOWORK='off', GOTOOLCHAIN='local', GOCACHE=root+'/gocache')
def regen():
    subprocess.check_call(['go','run','gen.go'],cwd=gdir,env=dict(env,GOFLAGS='-mod=mod'),stdout=subprocess.DEVNULL)
    for f in ['cmul_arm64.s','stockham_arm64.s','stockham32_arm64.s']: shutil.move(gdir+'/'+f, fft+'/internal/kernels/'+f)
try:
  for name,a,b in muts:
    c=tail.count(a)
    if c!=1: print('SKIP',name,c); continue
    open(gen,'w').write(head+tail.replace(a,b))
    regen()
    r=subprocess.run(['go','test','-run','Stockham32','.'],cwd=fft,env=env,capture_output=True,text=True)
    res='CAUGHT' if r.returncode!=0 else 'MISSED'
    which='whole' if 'TestStockham32PassMatchesScalar' in r.stdout and 'FAIL: TestStockham32PassMatchesScalar' in r.stdout else ''
    which+=' each' if 'FAIL: TestStockham32EachPassMatchesScalar' in r.stdout else ''
    print(name,res,which)
finally:
  open(gen,'w').write(orig); regen()
