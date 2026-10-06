# Batch-kernel mutations: (file, TEXT symbol, pattern, nth occurrence in that
# function, replacement). mode "build-amd64" builds a mutant test binary per
# mutation; mode "run-arm64" runs the strip/batch tests locally.
import subprocess,sys,os
S=os.environ['R25ROOT']  # holds fft/ (the clone), bin*/ and gocache/
mode=sys.argv[1]
A='internal/kernels/batch32_amd64.s'; N='internal/kernels/batch32_arm64.s'
muts={'amd64':[
 (A,'sk32Batch3AVX2','VBROADCASTSS 0(R10), Y9',1,'VBROADCASTSS 4(R10), Y9'),
 (A,'sk32Batch4AVX2','ADDQ adjout+88(FP), BX',1,''),
 (A,'sk32Batch2AVX2','TESTQ $1, R11',2,'TESTQ $0, R11'),
 (A,'sk32Batch3AVX2','ADDQ $16, R10',1,'ADDQ $8, R10'),
 (A,'sk32Batch2AVX2','VBROADCASTSS 4(R10), X6',1,'VBROADCASTSS 0(R10), X6'),
 (A,'sk32Batch5AVX2','VXORPS 0(R14), Y10, Y10',1,''),
 (A,'sk32Batch8AVX2','VMULPS 224(R14), Y8, Y8',1,''),
],'arm64':[
 (N,'sk32Batch2NEON','VLD2R.P 8(R14), [V28.S4, V29.S4]',1,'VLD2R.P 8(R14), [V29.S4, V30.S4]'),
 (N,'sk32Batch2NEON','ADD $8, R6, R6',1,''),
 (N,'sk32Batch3NEON','ADD $16, R26, R26',1,'ADD $8, R26, R26'),
 (N,'sk32Batch5NEONInv','VFMLS V29.S4',1,'VFMLA V29.S4'),
 (N,'sk32Batch4NEONInv','ADD R16<<1, R6, R6',1,'ADD R16, R6, R6'),
 (N,'sk32Batch8NEON','VFMUL V28.S4',2,'VFMUL V29.S4'),
]}[mode]
env=dict(os.environ)
if mode=='amd64': env.update(GOOS='linux',GOARCH='amd64')
for i,(f,sym,pat,nth,rep) in enumerate(muts,1):
    orig=open(f).read()
    start=orig.index('TEXT ·'+sym+'(SB)')
    end=orig.find('\nTEXT ',start+1); end=len(orig) if end<0 else end
    body=orig[start:end]; pos=-1
    for _ in range(nth):
        pos=body.index(pat,pos+1)
    body=body[:pos]+rep+body[pos+len(pat):]
    open(f,'w').write(orig[:start]+body+orig[end:])
    try:
        if mode=='amd64':
            subprocess.run(['go','test','-c','-ldflags=-s -w','-o',f'{S}/bin3/ftBM{i}.amd64','.'],check=True,env=env)
            print(i,sym,pat,'->',rep or '(removed)')
        else:
            r=subprocess.run(['go','test','-count=1','-run','Strips32','.'],capture_output=True,text=True,env=env)
            print('CAUGHT' if r.returncode else 'MISSED',i,sym,pat,'->',rep or '(removed)')
    finally:
        open(f,'w').write(orig)
