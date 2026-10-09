# acc31.py DIR: numpy's transform of the exact input bytes Go exported, against Go's old and new outputs.
import sys, os, numpy as np
d = sys.argv[1]
def rd(name, dt): return np.fromfile(os.path.join(d, name), dtype='<f8')
def c(x): return x.view('<c16')
def err(got, ref): return np.linalg.norm(got - ref) / np.linalg.norm(ref), np.max(np.abs(got - ref)) / np.max(np.abs(ref))
rows = []
for f in sorted(os.listdir(d)):
    if not f.startswith('in_'): continue
    name = f[3:-4]
    x = rd(f, 'f8')
    if name.startswith('nd_'):
        m, n = map(int, name[3:].split('x')); ref = np.fft.fft2(c(x).reshape(m, n)).ravel()
    elif name.startswith('c_'):
        ref = np.fft.fft(c(x))
    elif name.startswith('ir_'):
        n = int(name[3:]); ref = np.fft.irfft(c(x), n)
    elif name.startswith('r_'):
        ref = np.fft.rfft(x)
    for tag in ('old', 'new'):
        g = rd(f'{tag}_{name}.bin', 'f8')
        if not name.startswith('ir_'): g = c(g)
        rel, mx = err(g, ref)
        rows.append((name, tag, rel, mx))
for r in rows: print('%-14s %s  rel-l2 %.3e  max-rel %.3e' % r)
print('numpy', np.__version__)
