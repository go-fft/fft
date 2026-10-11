"""hw_acc.py DIR: numpy.fft on the exact input bytes TestHwExport wrote, against
go-fft's outputs; relative RMS and max error (to the largest output) per file.
With --plant, a byte in the high mantissa of one output value is changed
first, and the check must then fail (exit 1)."""
import glob, os, sys
import numpy as np

BOUND = 1e-14
d = sys.argv[1]
plant = "--plant" in sys.argv
print("numpy", np.__version__)
bad = 0
for inp in sorted(glob.glob(os.path.join(d, "in_*.bin"))):
    name = os.path.basename(inp)[3:-4]          # r_256, i_256, nd_1024x1024, c_6144
    kind, rest = name.split("_", 1)
    x = np.fromfile(inp, dtype="<f8")
    got = np.fromfile(os.path.join(d, "out_" + name + ".bin"), dtype="<f8")
    if kind == "r":
        ref = np.fft.rfft(x); got = got.view("<c16")
    elif kind == "i":
        n = int(rest); ref = np.fft.irfft(x.view("<c16"), n)
    elif kind == "c":
        ref = np.fft.fft(x.view("<c16")); got = got.view("<c16")
    elif kind == "nd":
        shape = tuple(int(s) for s in rest.split("x"))
        ref = np.fft.fftn(x.view("<c16").reshape(shape)).ravel(); got = got.view("<c16")
    else:
        continue
    if plant:
        b = bytearray(got.tobytes()); b[len(b) // 2 + 6] ^= 0x10
        got = np.frombuffer(bytes(b), dtype=got.dtype)
    scale = np.max(np.abs(ref))
    rms = np.sqrt(np.mean(np.abs(got - ref) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2))
    mx = np.max(np.abs(got - ref)) / scale
    ok = mx < BOUND
    bad += not ok
    print(f"{name:20s} rel_rms {rms:.2e}  max/scale {mx:.2e}  {'ok' if ok else 'FAIL'}")
print("FAILURES", bad)
sys.exit(1 if bad else 0)
