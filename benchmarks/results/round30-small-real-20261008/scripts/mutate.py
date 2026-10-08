# mutate.py: build one amd64 test binary per generator mutation of Round 30's
# untangle kernels (genSmallUntangle / smallUntangleStep in
# internal/kernels/asmgen/amd64/gen.go).
import os, shutil, subprocess, sys
S = os.path.dirname(os.path.abspath(__file__)) + "/.."
SRC = S + "/fft"
MUT = [
  ("xo sign on the imaginary lane", 'fmt.Sprintf("VXORPD Y13, %s, %s", b, b),', 'fmt.Sprintf("VXORPD Y15, %s, %s", b, b),'),
  ("wi read as wr", 'fmt.Sprintf("VMOVDDUP %d(CX), %s", o+8, d),', 'fmt.Sprintf("VMOVDDUP %d(CX), %s", o, d),'),
  ("xo not swapped", '\t\t\tfmt.Sprintf("VPERMILPD $5, %s, %s", b, b), // xo\n', ''),
  ("dst[m-k] as t - xe", 'fmt.Sprintf("VSUBPD %s, %s, %s", c, a, d),', 'fmt.Sprintf("VSUBPD %s, %s, %s", a, c, d),'),
  ("mirror not conjugated", 'fmt.Sprintf("VXORPD Y15, %s, %s", d, d),\n\t\t\t', ''),
  ("twiddle cursor half a step", 'Raw("ADDQ $%d, CX", 32*n)', 'Raw("ADDQ $%d, CX", 16*n)'),
  ("mirror offset of later steps", '"VMOVUPD %d(BX), %s", -o, b', '"VMOVUPD %d(BX), %s", o, b'),
  ("three steps with two pairs left", 'Raw("CMPQ R8, $%d", smallUntangleSteps).', 'Raw("CMPQ R8, $%d", smallUntangleSteps-1).'),
  ("tail step not advancing", '\tsmallAdvance(b, 1)\n', ''),
  ("retangle -wi as +wi", 'fmt.Sprintf("VXORPD Y12, %s, %s", c, c),', 'fmt.Sprintf("VXORPD Y13, %s, %s", c, c),'),
  ("retangle h replaced by 0.5", 'b.LoadArg("h", "R15").Raw("VMOVUPD (R15), Y14")', 'b.LoadArg("h", "R15").Raw("VMOVUPD 32(R14), Y14")'),
  ("retangle Z[k] as xe + [xoi,-xor]", 'fmt.Sprintf("VSUBPD %s, %s, %s", b, a, c),', 'fmt.Sprintf("VADDPD %s, %s, %s", b, a, c),'),
  ("two steps sharing a register", 'return fmt.Sprintf("Y%d", 4*j+i)', 'return fmt.Sprintf("Y%d", 4*(j%2)+i)'),
  ("product operands exchanged (NaN payload only; expected to survive)", 'fmt.Sprintf("VMULPD %s, %s, %s", b, d, d),', 'fmt.Sprintf("VMULPD %s, %s, %s", d, b, d),'),
]
env = dict(os.environ, GOWORK="off", GOTOOLCHAIN="local", GOCACHE=S + "/gocache", GOOS="linux", GOARCH="amd64")
for i, (name, old, new) in enumerate(MUT):
    d = S + "/mut/w%d" % i
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(SRC, d, ignore=shutil.ignore_patterns(".git"))
    g = d + "/internal/kernels/asmgen/amd64/gen.go"
    s = open(g).read()
    if s.count(old) < 1:
        print("MISSING", i, name); sys.exit(1)
    s = s.replace(old, new)
    open(g, "w").write(s)
    genv = dict(env); genv.pop("GOOS"); genv.pop("GOARCH"); genv["GOFLAGS"] = "-mod=mod"
    subprocess.run(["go", "run", "gen.go"], cwd=d + "/internal/kernels/asmgen/amd64", env=genv, check=True, stdout=subprocess.DEVNULL)
    shutil.move(d + "/internal/kernels/asmgen/amd64/untangle_amd64.s", d + "/internal/kernels/untangle_amd64.s")
    r = subprocess.run(["go", "test", "-c", "-ldflags=-s -w", "-o", S + "/mut/m%d.test" % i, "."], cwd=d, env=env)
    print(i, name, "built" if r.returncode == 0 else "BUILD FAILED", flush=True)
    shutil.rmtree(d)
