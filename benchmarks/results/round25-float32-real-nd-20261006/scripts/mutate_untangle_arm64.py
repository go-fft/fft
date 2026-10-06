import sys,subprocess,shutil
orig=open('/private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent3-f32/u32arm.s.orig').read()
tgt='internal/kernels/untangle32_arm64.s'
split=orig.index('TEXT ·f32rRetangleNEON')
U,R=orig[:split],orig[split:]
muts=[
 ('U: -(c-p) instead of (-c)+p','U',"VFNEG V12.S4, V14.S4\n\tVFMLA V31.S4, V6.S4, V14.S4","VORR V12.B16, V12.B16, V14.B16\n\tVFMLS V31.S4, V6.S4, V14.S4\n\tVFNEG V14.S4, V14.S4"),
 ('U: tr unfused','U',"VFMLS V11.S4, V5.S4, V12.S4","VFMUL V11.S4, V5.S4, V16.S4\n\tVFSUB V16.S4, V12.S4, V12.S4"),
 ('U: xe rounded first','U',"VFMLA V31.S4, V6.S4, V12.S4","VFMUL V31.S4, V6.S4, V16.S4\n\tVFADD V16.S4, V12.S4, V12.S4"),
 ('U: ti other product fused','U',"VFMUL V4.S4, V11.S4, V13.S4\n\tVFMLA V10.S4, V5.S4, V13.S4","VFMUL V5.S4, V10.S4, V13.S4\n\tVFMLA V11.S4, V4.S4, V13.S4"),
 ('R: mirror not reversed','R',"VEXT $8, V14.B16, V14.B16, V14.B16\n","" ),
 ('R: zk.im -(c-p)','R',"VFNEG V13.S4, V15.S4\n\tVFMLA V31.S4, V7.S4, V15.S4\n\tVFNEG V15.S4, V15.S4","VORR V13.B16, V13.B16, V15.B16\n\tVFMLS V31.S4, V7.S4, V15.S4"),
 ('R: dr unrounded (fused into xor)','R',"VFMUL V4.S4, V8.S4, V13.S4","VFMUL V5.S4, V8.S4, V13.S4"),
 ('R: zmk.re (-c)+p','R',"VFMLA V31.S4, V6.S4, V14.S4","VFMUL V31.S4, V6.S4, V16.S4\n\tVFADD V16.S4, V14.S4, V14.S4"),
]
for name,part,a,b in muts:
    src=U if part=='U' else R
    assert a in src,(name)
    s=(U.replace(a,b,1)+R) if part=='U' else (U+R.replace(a,b,1))
    open(tgt,'w').write(s)
    r=subprocess.run(['go','test','-count=1','-run','Untangle32Matches|Retangle32Matches','.'],capture_output=True,text=True)
    print('CAUGHT' if r.returncode!=0 else 'MISSED', name, (r.stdout+r.stderr).strip().splitlines()[0][:150] if r.returncode else '')
open(tgt,'w').write(orig)
