# Regenerates scipy_r2r.json, the reference vectors r2r_scipy_test.go checks
# DCT/IDCT/DST/IDST and their N-D forms against. Run with a Python that has
# scipy installed (the recorded file was made with the version it names):
#
#     python gen_scipy_r2r.py > scipy_r2r.json
#
# Tests never call Python; they only read the JSON.
import json

import numpy as np
import scipy
import scipy.fft as f


def signal(n, seed):
    i = np.arange(n, dtype=float)
    return np.sin(0.7 * i + seed) + 0.25 * ((i * 7 + seed) % 5) - 0.5


cases = []
for name, fn in (("dct", f.dct), ("idct", f.idct), ("dst", f.dst), ("idst", f.idst)):
    for typ in (1, 2, 3, 4):
        for norm in ("backward", "ortho", "forward"):
            for n in (1, 2, 3, 5, 8, 16, 17, 31):
                if name.endswith("dct") and typ == 1 and n == 1:
                    continue  # scipy rejects DCT-I of length 1
                x = signal(n, typ + n)
                y = fn(x, type=typ, norm=norm)
                cases.append({"fn": name, "type": typ, "norm": norm, "x": x.tolist(), "y": y.tolist()})

nd = []
for name, fn in (("dctn", f.dctn), ("idctn", f.idctn), ("dstn", f.dstn), ("idstn", f.idstn)):
    for typ in (1, 2, 3, 4):
        for norm in ("backward", "ortho", "forward"):
            shape = (3, 4, 5)
            x = signal(60, typ).reshape(shape)
            y = fn(x, type=typ, norm=norm)
            nd.append({"fn": name, "type": typ, "norm": norm, "shape": list(shape),
                       "x": x.ravel().tolist(), "y": y.ravel().tolist()})

json.dump({"scipy": scipy.__version__, "numpy": np.__version__, "cases": cases, "nd": nd},
          __import__("sys").stdout, indent=None, separators=(",", ":"))
