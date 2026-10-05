# Regenerates scipy_f32.json, the single-precision reference vectors
# scipy32_test.go checks the float32/complex64 N-D, Options and DCT/DST
# entry points against. Run with a Python that has scipy installed (the
# recorded file names the versions it was made with):
#
#     python gen_scipy_f32.py > scipy_f32.json
#
# Every input is float32 or complex64, and every output is what scipy.fft
# returns for it, which stays float32/complex64 (the script asserts it).
# Complex arrays are stored interleaved: re0, im0, re1, im1, ...
# Tests never call Python; they only read the JSON.
import json
import sys

import numpy as np
import scipy
import scipy.fft as f


def real_signal(n, seed):
    i = np.arange(n, dtype=np.float64)
    return (np.sin(0.7 * i + seed) + 0.25 * ((i * 7 + seed) % 5) - 0.5).astype(np.float32)


def complex_signal(n, seed):
    return (real_signal(n, seed) + 1j * real_signal(n, seed + 11)).astype(np.complex64)


def flat(a):
    # Each float32 is written as the shortest decimal that rounds back to it
    # (numpy's unique repr), which a reader recovers exactly by parsing the
    # number and rounding it to float32.
    a = np.asarray(a).ravel()
    if np.iscomplexobj(a):
        assert a.dtype == np.complex64, a.dtype
        a = np.column_stack((a.real, a.imag)).ravel()
    else:
        assert a.dtype == np.float32, a.dtype
    return [float(np.format_float_positional(v, unique=True)) for v in a]


def axes_arg(axes):
    return None if axes is None else tuple(axes)


cases = []
norms = ("backward", "ortho", "forward")

# Complex N-D, with and without axes.
for shape, axes in (((3, 4, 5), None), ((3, 4, 5), [2, 0]), ((6, 7), None), ((6, 7), [-1]), ((17,), None)):
    x = complex_signal(int(np.prod(shape)), len(shape)).reshape(shape)
    for norm in norms:
        for name, fn in (("fftn", f.fftn), ("ifftn", f.ifftn)):
            y = fn(x, axes=axes_arg(axes), norm=norm)
            cases.append({"fn": name, "norm": norm, "shape": list(shape), "axes": axes, "x": flat(x), "y": flat(y)})

# Real N-D, with and without axes; the inverse takes the forward's output.
for shape, axes in (((4, 6), None), ((3, 5, 8), None), ((3, 5, 8), [2, 0]), ((5, 9), [0]), ((7,), None)):
    x = real_signal(int(np.prod(shape)), 3).reshape(shape)
    for norm in norms:
        y = f.rfftn(x, axes=axes_arg(axes), norm=norm)
        cases.append({"fn": "rfftn", "norm": norm, "shape": list(shape), "axes": axes, "x": flat(x), "y": flat(y)})
        s = [shape[a] for a in (axes if axes is not None else range(len(shape)))]
        z = f.irfftn(y, s=s, axes=axes_arg(axes), norm=norm)
        cases.append({"fn": "irfftn", "norm": norm, "shape": list(shape), "axes": axes, "x": flat(y), "y": flat(z)})

# 1-D with n.
for n_in, n in ((10, None), (10, 16), (10, 7), (9, None)):
    xc = complex_signal(n_in, 5)
    xr = real_signal(n_in, 6)
    for norm in norms:
        for name, fn, x in (("fft", f.fft, xc), ("ifft", f.ifft, xc), ("rfft", f.rfft, xr), ("ihfft", f.ihfft, xr),
                            ("irfft", f.irfft, xc), ("hfft", f.hfft, xc)):
            y = fn(x, n=n, norm=norm)
            cases.append({"fn": name, "norm": norm, "n": 0 if n is None else n, "x": flat(x), "y": flat(y)})

# DCT/DST, 1-D and N-D.
for name, fn in (("dct", f.dct), ("idct", f.idct), ("dst", f.dst), ("idst", f.idst)):
    for typ in (1, 2, 3, 4):
        for norm in norms:
            for n in (2, 5, 8, 17):
                x = real_signal(n, typ + n)
                y = fn(x, type=typ, norm=norm)
                cases.append({"fn": name, "type": typ, "norm": norm, "x": flat(x), "y": flat(y)})
for name, fn in (("dctn", f.dctn), ("idctn", f.idctn), ("dstn", f.dstn), ("idstn", f.idstn)):
    for typ in (1, 2, 3, 4):
        for norm in norms:
            shape = (3, 4, 5)
            x = real_signal(60, typ).reshape(shape)
            y = fn(x, type=typ, norm=norm)
            cases.append({"fn": name, "type": typ, "norm": norm, "shape": list(shape), "x": flat(x), "y": flat(y)})

json.dump({"scipy": scipy.__version__, "numpy": np.__version__, "cases": cases},
          sys.stdout, indent=None, separators=(",", ":"))
