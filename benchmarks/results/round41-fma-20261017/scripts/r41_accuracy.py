# r41_accuracy.py DIR: numpy.fft on the exact input bytes TestR41Export wrote,
# against go-fft's default and FMA outputs; also scipy.fft in long double
# (80-bit on x86-64) as a reference more exact than either.
import sys, os, numpy as np
d = sys.argv[1]
try:
    import scipy.fft as sf
except ImportError:
    sf = None
def rd(name): return np.fromfile(os.path.join(d, name), dtype='<f8')
def c(x): return x.view('<c16')
def rel(got, ref): return float(np.sqrt(np.mean(np.abs(got - ref) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2)))
def ref64(kind, x, name):
    if kind == 'c': return np.fft.fft(c(x))
    if kind == 'ic': return np.fft.ifft(c(x))
    if kind == 'r': return np.fft.rfft(x)
    if kind == 'ir': return np.fft.irfft(c(x), int(name))
    m, n = map(int, name.split('x')); return np.fft.fft2(c(x).reshape(m, n)).ravel()
def refld(kind, x, name):
    if sf is None: return None
    L = np.longdouble
    if kind in ('c', 'ic', 'n'): z = c(x).real.astype(L) + 1j * c(x).imag.astype(L)
    if kind == 'c': return sf.fft(z)
    if kind == 'ic': return sf.ifft(z)
    if kind == 'r': return sf.rfft(x.astype(L))
    if kind == 'ir':
        z = c(x).real.astype(L) + 1j * c(x).imag.astype(L); return sf.irfft(z, int(name))
    m, n = map(int, name.split('x')); return sf.fft2(z.reshape(m, n)).ravel()
print('%-6s %-10s %12s %12s %12s %12s %8s' % ('kind', 'n', 'def/numpy', 'fma/numpy', 'def/ld', 'fma/ld', 'fma÷def'))
worst = 0.0
for f in sorted(os.listdir(d)):
    if not f.startswith('in_'): continue
    kind, name = f[3:-4].split('_', 1)
    for k in ([kind] if kind != 'c' else ['c', 'ic']):
        x = rd(f)
        if kind == 'ir': pass
        g = {}
        for tag in ('def', 'fma'):
            v = rd('%s_%s_%s.bin' % (tag, k, name))
            g[tag] = v if k == 'ir' else c(v)
        r = ref64(k, x, name)
        rl = refld(k, x, name)
        e = [rel(g['def'], r), rel(g['fma'], r)]
        if rl is not None:
            e += [rel(g['def'].astype(rl.dtype), rl), rel(g['fma'].astype(rl.dtype), rl)]
            ratio = e[3] / e[2] if e[2] > 0 else float('nan')
        else:
            e += [float('nan')] * 2; ratio = e[1] / e[0]
        worst = max(worst, ratio)
        print('%-6s %-10s %12.3e %12.3e %12.3e %12.3e %8.3f' % (k, name, e[0], e[1], e[2], e[3], ratio))
print('worst fma÷def (long-double reference):', '%.3f' % worst)
print('numpy', np.__version__, 'scipy', sf and __import__('scipy').__version__)
