import sys, json
import numpy as np

# Value parity: for each {"name", "setup", "call"} on stdin, evaluate the same
# setup and call the timing worker evaluates (pyworker.py) and print the result
# as one flat list of floats (complex values interleaved), for the Go test to
# compare at a tolerance.
env = {"np": np}
out = {}
for c in json.load(sys.stdin):
    args = eval(c["setup"], dict(env))
    f = eval(c["call"], dict(env))
    r = np.ascontiguousarray(np.asarray(f(*args)))
    if np.iscomplexobj(r):
        r = r.view(r.real.dtype)
    out[c["name"]] = [float(v) for v in r.ravel()]
json.dump(out, sys.stdout)
