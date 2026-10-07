#!/usr/bin/env python3
"""Mutation test of genArmrealUntangle: apply each mutation to gen.go,
regenerate untangle_arm64.s, run the untangle tests, restore."""
import subprocess,shutil,os,sys
F=os.environ['F']; G=F+'/internal/kernels/asmgen/arm64'
gen=open(G+'/gen.go').read(); sfile=F+'/internal/kernels/untangle_arm64.s'; s0=open(sfile).read()
i0=gen.index('func armrealStep(')
M=[
 ("untangle tr unfused", "v3(fmls, 2, 5, 1),", "v3(mul, 9, 5, 1), v3(sub, 2, 2, 9),"),
 ("untangle xe halving rounded first", "v3(fmla, 2, 6, h), // dst[k].re", "v3(mul, 4, 6, h), v3(add, 2, 2, 4), // dst[k].re"),
 ("untangle ti fuses the other product", "v3(mul, 3, 1, 4),  // wr·xoi\n\t\t\tv3(fmla, 3, 5, 0),", "v3(mul, 3, 5, 0),  // wr·xoi\n\t\t\tv3(fmla, 3, 1, 4),"),
 ("mirror input not swapped", "swap(2), swap(3),", "swap(2),"),
 ("mirror output not swapped", "swap(mr), swap(mi),", "swap(mr),"),
 ("untangle -(c - p) form", "neg(8, 2),         //\n\t\t\tv3(fmla, 8, 6, h),", "v3(mul, 8, 6, h), v3(sub, 8, 2, 8), neg(8, 8),"),
 ("untangle xoi sign", "neg(1, 1),         // xoi", ""),
 ("retangle xor fuses the other product", "v3(mul, 3, 8, 4),  // wr·dr\n\t\t\tv3(fmla, 3, 9, 5),", "v3(mul, 3, 9, 5),  // wr·dr\n\t\t\tv3(fmla, 3, 8, 4),"),
 ("retangle h·s.re rounded first", "v3(fmla, 0, 6, h), // z[m-k].re", "v3(mul, 4, 6, h), v3(add, 0, 0, 4), // z[m-k].re"),
 ("retangle xoi fuses the other product", "v3(mul, 0, 9, 4),  // wr·di\n\t\t\tv3(fmls, 0, 8, 5),", "v3(mul, 0, 8, 5),  // wr·di\n\t\t\tneg(0, 0), v3(fmla, 0, 9, 4),"),
 ("retangle z[m-k].im as VFMLS", "neg(1, 3),         //\n\t\t\tv3(fmla, 1, 7, h), // h·d.im − xor\n\t\t\tneg(1, 1),", "raw(\"VMOV V%d.B16, V%d.B16\", 3, 1), v3(fmls, 1, 7, h),"),
 ("mirror pointer step", "Raw(\"SUB $%d, R8, R8\", 32*u)", "Raw(\"SUB $%d, R8, R8\", 16*u)"),
 ("twiddle not advanced", "raw(\"VLD2.P 32(\"+tw+\"), [V%d.D2, V%d.D2]\", 4, 5)", "raw(\"VLD2 (\"+tw+\"), [V%d.D2, V%d.D2]\", 4, 5)"),
 ("first bin offset", "Raw(\"ADD $16, R2, R9\")", "Raw(\"ADD $32, R2, R9\")"),
 ("steps overlap their registers", "all[s][i](b, 10*s)", "all[s][i](b, 9*s)"),
 ("a tail step skipped", "Raw(\"CMP $%d, R4\", t)", "Raw(\"CMP $%d, R4\", t+1)"),
 ("second step's mirror offset", "Raw(\"SUB $%d, R6, %s\", 32*s, mirs[s][0])", "Raw(\"SUB $%d, R6, %s\", 32*s+32, mirs[s][0])"),
]
res=[]
for name,a,b in M:
    body=gen[i0:]
    if a not in body: print("NOT FOUND", name); continue
    g=gen[:i0]+body.replace(a,b,1)
    if 'VMOVQ2' in g:
        g=g.replace('VMOVQ2(15, 13).VFMLS2D(15, 7, 31).','Raw("VMOV V13.B16, V15.B16").VFMLS2D(15, 7, 31).')
        # Raw returns *Builder so the chain continues
    open(G+'/gen.go','w').write(g)
    try:
        r=subprocess.run('cd %s && GOFLAGS=-mod=mod go run gen.go >/dev/null && mv untangle_arm64.s ../../ && rm -f *.s'%G,shell=True,capture_output=True,text=True)
        if r.returncode: res.append((name,'GEN FAIL '+r.stderr[-300:])); continue
        t=subprocess.run('cd %s && go test -count=1 -run "UntangleMatchesScalarNEON|RetangleMatchesScalarNEON|RealPlanUntangleNEONEndToEnd" . 2>&1 | tail -3'%F,shell=True,capture_output=True,text=True)
        res.append((name,'CAUGHT' if 'FAIL' in t.stdout else 'SURVIVED: '+t.stdout))
    finally:
        open(G+'/gen.go','w').write(gen); open(sfile,'w').write(s0)
for n,r in res: print(f"{n:40s} {r}")
