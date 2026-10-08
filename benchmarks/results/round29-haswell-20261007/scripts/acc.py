"""acc.py DIR: for each n, the forward transform of Go's own input bytes
(in_n.bin) by numpy.fft, against Go's two outputs (cur_n.bin: main's
factorization, c2_n.bin: Round 29's); relative RMS and max error."""
import numpy as np, sys, glob, re, os
d=sys.argv[1]
print(f"numpy {np.__version__}")
print(f"{'n':>8} {'cur rms':>10} {'new rms':>10} {'cur max':>10} {'new max':>10}  new/cur rms")
for f in sorted(glob.glob(os.path.join(d,'in_*.bin')), key=lambda p:int(re.findall(r'\d+',os.path.basename(p))[0])):
    n=int(re.findall(r'\d+',os.path.basename(f))[0])
    x=np.fromfile(f,dtype='<c16'); assert len(x)==n
    ref=np.fft.fft(x); nr=np.linalg.norm(ref)
    out=[]
    for tag in ('cur','c2'):
        y=np.fromfile(os.path.join(d,f'{tag}_{n}.bin'),dtype='<c16')
        e=y-ref
        out.append((np.linalg.norm(e)/nr, np.max(np.abs(e))/np.max(np.abs(ref))))
    print(f"{n:8d} {out[0][0]:10.3e} {out[1][0]:10.3e} {out[0][1]:10.3e} {out[1][1]:10.3e}  {out[1][0]/out[0][0]:.3f}")
