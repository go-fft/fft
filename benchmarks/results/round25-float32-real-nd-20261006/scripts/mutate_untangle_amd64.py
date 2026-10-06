import subprocess,os
S=os.environ['R25ROOT']  # holds fft/ (the clone), bin*/ and gocache/
tgt='internal/kernels/untangle32_amd64.s'
orig=open(tgt).read()
split=orig.index('TEXT ·f32rRetangleAVX2')
U,R=orig[:split],orig[split:]
muts=[
 ('U','VPERMPD $0x1B, Y11, Y11','VPERMPD $0x4E, Y11, Y11'),
 ('U','VADDSUBPS Y8, Y7, Y7','VADDSUBPS Y7, Y8, Y7'),
 ('U','VBLENDPS $0xAA, Y3, Y2, Y4','VBLENDPS $0x55, Y3, Y2, Y4'),
 ('U','VXORPS Y12, Y5, Y5\n',''),
 ('R','VXORPS Y13, Y6, Y6\n',''),
 ('R','VMOVSLDUP Y6, Y7\n\tVMOVSHDUP Y6, Y8','VMOVSHDUP Y6, Y7\n\tVMOVSLDUP Y6, Y8'),
 ('R','VXORPS Y13, Y11, Y11\n',''),
]
env=dict(os.environ,GOOS='linux',GOARCH='amd64')
for i,(part,a,b) in enumerate(muts,1):
    src=U if part=='U' else R
    assert a in src,a
    s=(U.replace(a,b,1)+R) if part=='U' else (U+R.replace(a,b,1))
    open(tgt,'w').write(s)
    subprocess.run(['go','test','-c','-ldflags=-s -w','-o',f'{S}/bin/ftM{i}.amd64','.'],check=True,env=env)
    print(i,part,a.strip().replace('\n',' / '),'->',b.strip().replace('\n',' / '))
open(tgt,'w').write(orig)
