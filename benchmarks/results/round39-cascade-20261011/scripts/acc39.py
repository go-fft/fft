# acc39.py DIR: numpy's transform of the exact input bytes Go exported (TestCl3Export), against Go's old and new outputs.
import sys, os, numpy as np
d = sys.argv[1]
def rd(name): return np.fromfile(os.path.join(d, name), dtype='<f8')
def c(x): return x.view('<c16')
def err(got, ref): return np.linalg.norm(got - ref) / np.linalg.norm(ref), np.max(np.abs(got - ref)) / np.max(np.abs(ref))
rows, same = [], []
for f in sorted(os.listdir(d)):
    if not f.startswith('in_'): continue
    name = f[3:-4]
    kind, shp = name.split('_', 1)
    shape = tuple(map(int, shp.split('x')))
    x = rd(f)
    if kind == 'nd':
        ref = np.fft.fftn(c(x).reshape(shape)).ravel()
    elif kind == 'ind':
        ref = np.fft.ifftn(c(x).reshape(shape)).ravel()
    elif kind == 'r2':
        ref = np.fft.rfft2(x.reshape(shape)).ravel()
    out = {}
    for tag in ('old', 'new'):
        g = c(rd(f'{tag}_{name}.bin'))
        out[tag] = g
        rel, mx = err(g, ref)
        rows.append((name, tag, rel, mx))
    same.append((name, np.array_equal(out['old'].view('<u8'), out['new'].view('<u8'))))
for r in rows: print('%-16s %s  rel-l2 %.3e  max-rel %.3e' % r)
for s in same: print('%-16s old and new bit-identical: %s' % s)
worst = max(r[3] for r in rows)
print('worst max-rel %.3e (bound 1e-14): %s' % (worst, 'ok' if worst < 1e-14 else 'FAIL'))
print('numpy', np.__version__)
# Planted fault: one high mantissa byte of the new 512x512 output changed must fail the bound.
g = rd('new_nd_512x512.bin').copy()
b = g.view('u1'); b[8 * 1000 + 5] ^= 0x40
ref = np.fft.fftn(c(rd('in_nd_512x512.bin')).reshape(512, 512)).ravel()
print('planted byte in new_nd_512x512: max-rel %.3e -> %s' % (err(c(g), ref)[1], 'caught' if err(c(g), ref)[1] >= 1e-14 else 'MISSED'))
