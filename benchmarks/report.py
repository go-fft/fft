#!/usr/bin/env python3
"""Consolidate the four measured data sources into the standardized parity
report BENCHMARKS.md.

Inputs (all produced by run.sh on the same host):
  - go_bench.txt    : `go test -bench` output for go-fft + gonum (ns/op)
  - go_plan.txt     : go-fft / gonum plan-construction ns/op
  - fftw.json       : native-FFTW C harness (ns/op, GFLOP/s, plan ns)
  - ref.json        : numpy.fft + scipy.fft (pocketfft) ns/op + GFLOP/s

Emits BENCHMARKS.md with the parity table (GFLOP/s + ns/op), go/FFTW ratio,
verdicts, and the lagging-ops action items. No fabrication: every cell is a
measured number or "—" when an implementation does not cover that case.
"""
import json
import os
import math
import re
import sys

# How the FFTW the C harness linked was built; run.sh's default is the Homebrew
# bottle. Set FFTW_DESC when the harness runs against another build.
FFTW_DESC = os.environ.get("FFTW_DESC", "Homebrew arm64 bottle, NEON, linked from C")

HOST = {
    "machine": "Apple M4 Max (16-core: 12P+4E), macOS 26.5 (25F71), arm64",
    "go": "go1.26.4 darwin/arm64",
    "fftw": None,            # filled from fftw.json
    "numpy": None, "scipy": None, "pyfftw": None,
    "gonum": "gonum.org/v1/gonum v0.16.0",
}


def gflops(n, ns, real=False):
    if not ns or ns <= 0 or n <= 1:
        return 0.0
    f = 5.0 * n * math.log2(n)
    if real:
        f *= 0.5
    return f / (ns * 1e-9) / 1e9


def parse_go(path):
    """Return {bench_base: {size_key: ns}} from go test -bench output."""
    out = {}
    pat = re.compile(r"^Benchmark(\w+)/([\w x]+)-\d+\s+\d+\s+([\d.]+)\s+ns/op")
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            base, size, ns = m.group(1), m.group(2), float(m.group(3))
            d = out.setdefault(base, {})
            # Keep the best (minimum) across repeated -count runs, matching the
            # best-of-N convention used for the FFTW/numpy/scipy harnesses.
            d[size] = ns if size not in d else min(d[size], ns)
    return out


def fmt_ns(ns):
    if ns is None:
        return "—"
    return f"{ns:,.0f}"


def fmt_g(g):
    return "—" if g is None else f"{g:.1f}"


def main():
    go = parse_go("go_bench.txt")
    goplan = parse_go("go_plan.txt")
    fftw = json.load(open("fftw.json"))
    ref = json.load(open("ref.json"))
    HOST["fftw"] = fftw.get("fftw_version")
    HOST["numpy"] = ref["versions"]["numpy"]
    HOST["scipy"] = ref["versions"]["scipy"]
    HOST["pyfftw"] = ref["versions"]["pyfftw"]

    fftw_c = {r["n"]: r for r in fftw["complex"]}
    fftw_r = {r["n"]: r for r in fftw["real"]}
    fftw_2 = {r["shape"]: r for r in fftw["fft2"]}
    ref_c = {r["n"]: r for r in ref["complex"]}
    ref_r = {r["n"]: r for r in ref["real"]}
    ref_2 = {r["shape"]: r for r in ref["fft2"]}

    L = []
    w = L.append

    w("# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum\n")
    w("Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, "
      "measured against the gold-standard C library **FFTW** and the reference "
      "Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all "
      "on the **same machine, same inputs, same sizes**.\n")
    w("> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the "
      "native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then "
      "rebuilds this file). Numerical correctness is gated first: every go-fft "
      "transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` "
      "before any timing is reported.\n")

    w("## Methodology\n")
    w(f"- **Machine**: {HOST['machine']}.")
    w(f"- **Toolchains**: {HOST['go']}; native **FFTW {HOST['fftw']}** "
      f"({FFTW_DESC}); "
      f"**numpy {HOST['numpy']}** / **scipy {HOST['scipy']}** (pocketfft); "
      f"**pyfftw {HOST['pyfftw']}**; **{HOST['gonum']}**.")
    w("- **Single-threaded** for the apples-to-apples core comparison: FFTW "
      "planned with `threads=1`; numpy/scipy pinned via `OMP_NUM_THREADS=1 "
      "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`; scipy "
      "`workers=1`; Go benchmarks are single-goroutine for 1-D. The 2-D rows are "
      "the one place go-fft uses its multicore path (the others are all 1-core).")
    w("- **Plan reuse / steady state**: go-fft via its cached `Plan` API "
      "(`NewPlan(n).FFT`, `NewRealPlan(n).RFFT`); gonum via its reused "
      "`CmplxFFT`/`FFT` object; FFTW via a reused `FFTW_MEASURE` plan; scipy via "
      "its internal plan cache. Each number is the **steady-state transform** "
      "cost, not planning — plan/setup cost is reported separately below.")
    w("- **Iterations**: Go uses `-benchtime=1s` (auto-scaled `b.N`); the C and "
      "Python harnesses auto-scale each batch to ~0.2 s and take the **best of "
      "6** batches after warm-up. Lower ns/op is better.")
    w("- **Metric**: ns/op and **GFLOP/s** using the standard `5·N·log2(N)` flop "
      "convention for a complex N-point FFT (real rfft counted at half, "
      "`2.5·N·log2(N)`; 2-D at `5·N·log2(N)` with N = total points).")
    w("- **Inputs**: bit-identical across all four implementations "
      "(`((i·7+1)%13)·0.1 + i·((i·3+2)%11)·0.1` for complex; "
      "`((i·7+1)%13)·0.1` for real).")
    w("- **Note on pyfftw**: the FFTW column is the **native FFTW called "
      "directly from C** (`benchmarks/cbench/fftw_bench.c`), not pyfftw: the "
      "pip-wheel FFTW bundled with pyfftw planned a 2-D transform about 4× "
      "slower than the native library on Apple Silicon.\n")

    def verdict(gns, fns):
        if gns is None or fns is None:
            return "—"
        r = gns / fns
        if r <= 1.05:
            return "**≥ parity**"
        return f"lags FFTW {r:.2f}×"

    # ---------------- Complex 1-D ----------------
    w("## Complex 1-D FFT (`complex128`)\n")
    w("ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW (lower is better; ≤1.05 = parity).\n")
    w("| N | go-fft | FFTW | numpy.fft | scipy.fft | gonum | go/FFTW | verdict |")
    w("|---:|---:|---:|---:|---:|---:|---:|:--|")
    gC = go.get("Complex_GoFFT", {})
    gN = go.get("Complex_Gonum", {})
    for n in [256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920, 1009, 1296, 10007]:
        k = str(n)
        gns = gC.get(k)
        fr = fftw_c.get(n, {})
        rr = ref_c.get(n, {})
        nns, sns = rr.get("numpy_ns"), rr.get("scipy_ns")
        gonm = gN.get(k)
        fns = fr.get("fftw_ns")
        tag = label_c(n)
        w(f"| {n:,}{tag} | {fmt_ns(gns)} ({gflops(n, gns):.1f}) "
          f"| {fmt_ns(fns)} ({fmt_g(fr.get('fftw_gflops'))}) "
          f"| {fmt_ns(nns)} ({gflops(n, nns):.1f}) "
          f"| {fmt_ns(sns)} ({gflops(n, sns):.1f}) "
          f"| {fmt_ns(gonm)} ({gflops(n, gonm):.1f}) "
          f"| {(gns/fns if gns and fns else float('nan')):.2f}× "
          f"| {verdict(gns, fns)} |")
    w("")

    # ---------------- Real 1-D ----------------
    w("## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)\n")
    w("| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |")
    w("|---:|---:|---:|---:|---:|---:|---:|:--|")
    gC = go.get("Real_GoFFT", {})
    gN = go.get("Real_Gonum", {})
    for n in [256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920]:
        k = str(n)
        gns = gC.get(k)
        fr = fftw_r.get(n, {})
        rr = ref_r.get(n, {})
        nns, sns = rr.get("numpy_ns"), rr.get("scipy_ns")
        gonm = gN.get(k)
        fns = fr.get("fftw_ns")
        tag = label_c(n)
        w(f"| {n:,}{tag} | {fmt_ns(gns)} ({gflops(n, gns, True):.1f}) "
          f"| {fmt_ns(fns)} ({fmt_g(fr.get('fftw_gflops'))}) "
          f"| {fmt_ns(nns)} ({gflops(n, nns, True):.1f}) "
          f"| {fmt_ns(sns)} ({gflops(n, sns, True):.1f}) "
          f"| {fmt_ns(gonm)} ({gflops(n, gonm, True):.1f}) "
          f"| {(gns/fns if gns and fns else float('nan')):.2f}× "
          f"| {verdict(gns, fns)} |")
    w("")

    # ---------------- Real inverse 1-D (c2r) ----------------
    fftw_cr = {r["n"]: r for r in fftw.get("creal", [])}
    if fftw_cr:
        w("## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)\n")
        w("The c2r inverse mirrors the forward packing: an N/2-point inverse "
          "complex FFT plus an untangle pass, half the work of a full length-N "
          "conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.\n")
        w("| N | go-fft | FFTW | go/FFTW | verdict |")
        w("|---:|---:|---:|---:|:--|")
        gI = go.get("CReal_GoFFT", {})
        for n in [256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920]:
            k = str(n)
            gns = gI.get(k)
            fr = fftw_cr.get(n, {})
            fns = fr.get("fftw_ns")
            tag = label_c(n)
            w(f"| {n:,}{tag} | {fmt_ns(gns)} ({gflops(n, gns, True):.1f}) "
              f"| {fmt_ns(fns)} ({fmt_g(fr.get('fftw_gflops'))}) "
              f"| {(gns/fns if gns and fns else float('nan')):.2f}× "
              f"| {verdict(gns, fns)} |")
        w("")

    # ---------------- 2-D ----------------
    w("## 2-D complex FFT2 (`complex128`)\n")
    w("go-fft fans the independent 1-D row/column transforms across goroutines "
      "above a work-size threshold (the multicore path); FFTW / numpy / scipy "
      "are single-threaded here. The go-fft column is a reused `PlanN` writing "
      "into a reused slice, as FFTW's plan writes into its reused array; the "
      "`FFT2` column is the convenience call, which returns a new slice each "
      "time and so also pays an allocation (and, on a many-core host, the "
      "garbage collector). ns/op (GFLOP/s).\n")
    w("| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |")
    w("|:--|---:|---:|---:|---:|---:|---:|:--|")
    g2 = go2d(go)
    g2a = go.get("FFT2_GoFFT", {})
    for shp in ["64x64", "128x128", "256x256", "512x512", "1024x1024"]:
        a, b = (int(v) for v in shp.split("x"))
        n = a * b
        gns = g2.get(shp)
        fr = fftw_2.get(shp, {})
        rr = ref_2.get(shp, {})
        nns, sns = rr.get("numpy_ns"), rr.get("scipy_ns")
        fns = fr.get("fftw_ns")
        ans = g2a.get(shp)
        w(f"| {shp} | {fmt_ns(gns)} ({gflops(n, gns):.1f}) "
          f"| {fmt_ns(ans)} ({gflops(n, ans):.1f}) "
          f"| {fmt_ns(fns)} ({fmt_g(fr.get('fftw_gflops'))}) "
          f"| {fmt_ns(nns)} ({gflops(n, nns):.1f}) "
          f"| {fmt_ns(sns)} ({gflops(n, sns):.1f}) "
          f"| {(gns/fns if gns and fns else float('nan')):.2f}× "
          f"| {verdict(gns, fns)} |")
    w("")

    # ---------------- Plan / setup cost ----------------
    w("## Plan / setup cost (built once, then amortized)\n")
    w("Steady-state transforms above reuse a plan. This is the one-time "
      "construction cost (ns), reported separately. go-fft and gonum build "
      "twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to "
      "pick codelets, so its planning is orders of magnitude more expensive — "
      "the price of its steady-state speed, paid back only across many reuses.\n")
    w("| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |")
    w("|---:|---:|---:|---:|")
    gp = goplan.get("ComplexPlan_GoFFT", {})
    np_ = goplan.get("ComplexPlan_Gonum", {})
    for n in [256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920, 1009, 1296, 10007]:
        k = str(n)
        fp = fftw_c.get(n, {}).get("fftw_plan_ns")
        w(f"| {n:,} | {fmt_ns(gp.get(k))} | {fmt_ns(np_.get(k))} | {fmt_ns(fp)} |")
    w("")

    # ---------------- Summary ----------------
    emit_summary(w, go, fftw_c, fftw_r, fftw_2, ref_c, ref_r, ref_2)

    with open("../BENCHMARKS.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote ../BENCHMARKS.md", file=sys.stderr)


def go2d(go):
    """The go-fft 2-D timings the report compares: the reused-plan benchmark
    when the run has it, else (older runs) the allocating FFT2 one."""
    return go.get("FFT2Plan_GoFFT") or go.get("FFT2_GoFFT", {})


def label_c(n):
    tags = {
        256: " (2⁸)", 1024: " (2¹⁰)", 4096: " (2¹²)", 65536: " (2¹⁶)",
        1048576: " (2²⁰)", 1000: " (2³·5³)", 1080: " (2³·3³·5)",
        1920: " (2⁷·3·5)", 1009: " (prime)", 1296: " (2⁴·3⁴)",
        10007: " (prime)",
    }
    return tags.get(n, "")


def emit_summary(w, go, fftw_c, fftw_r, fftw_2, ref_c, ref_r, ref_2):
    # Tally parity vs FFTW / numpy / scipy / gonum across complex+real+2d.
    def tally(go_base, fmap, refmap, sizes, gonum_base=None, real=False):
        res = {"fftw": [0, 0], "numpy": [0, 0], "scipy": [0, 0], "gonum": [0, 0]}
        gb = go.get(go_base, {})
        nb = go.get(gonum_base, {}) if gonum_base else {}
        for n in sizes:
            k = str(n)
            g = gb.get(k)
            if g is None:
                continue
            f = fmap.get(n, {}).get("fftw_ns")
            r = refmap.get(n, {})
            for ref_key, val in (("numpy", r.get("numpy_ns")),
                                 ("scipy", r.get("scipy_ns")),
                                 ("fftw", f),
                                 ("gonum", nb.get(k))):
                if val is None:
                    continue
                res[ref_key][1] += 1
                if g <= val * 1.05:
                    res[ref_key][0] += 1
        return res

    cs = [256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920, 1009, 1296, 10007]
    rs = [256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920]
    tc = tally("Complex_GoFFT", fftw_c, ref_c, cs, "Complex_Gonum")
    tr = tally("Real_GoFFT", fftw_r, ref_r, rs, "Real_Gonum", real=True)

    # 2-D tally
    t2 = {"fftw": [0, 0], "numpy": [0, 0], "scipy": [0, 0]}
    g2 = go2d(go)
    for shp in ["64x64", "128x128", "256x256", "512x512", "1024x1024"]:
        g = g2.get(shp)
        if g is None:
            continue
        r = ref_2.get(shp, {})
        for key, val in (("fftw", fftw_2.get(shp, {}).get("fftw_ns")),
                         ("numpy", r.get("numpy_ns")), ("scipy", r.get("scipy_ns"))):
            if val is None:
                continue
            t2[key][1] += 1
            if g <= val * 1.05:
                t2[key][0] += 1

    def merge(*ts):
        m = {}
        for t in ts:
            for k, (a, b) in t.items():
                if k not in m:
                    m[k] = [0, 0]
                m[k][0] += a
                m[k][1] += b
        return m

    allt = merge(tc, tr, t2)
    w("## Summary\n")
    w("At-or-above parity (≤ 1.05× the reference's ns/op) across all "
      "complex-1-D + real-1-D + 2-D rows measured on this host:\n")
    for key, label in (("fftw", "FFTW (native C, gold standard)"),
                       ("numpy", "numpy.fft (pocketfft)"),
                       ("scipy", "scipy.fft (pocketfft)"),
                       ("gonum", "gonum (pure-Go peer)")):
        a, b = allt.get(key, [0, 0])
        w(f"- **vs {label}**: {a}/{b} ops at-or-above parity.")
    w("")
    # Every row behind FFTW, worst first. What explains each band, and what
    # was tried against it, is in the dated rounds of BENCHMARKS.md: prose
    # written here once went stale as the engine changed, so the report now
    # states only what this run measured.
    lag = []
    for base, fmap, sizes, kind in (("Complex_GoFFT", fftw_c, cs, "complex"),
                                    ("Real_GoFFT", fftw_r, rs, "real")):
        gb = go.get(base, {})
        for n in sizes:
            g, f = gb.get(str(n)), fmap.get(n, {}).get("fftw_ns")
            if g and f and g > f * 1.05:
                lag.append((g / f, f"{kind} {n:,}{label_c(n)}"))
    g2 = go2d(go)
    for shp in ["64x64", "128x128", "256x256", "512x512", "1024x1024"]:
        g, f = g2.get(shp), fftw_2.get(shp, {}).get("fftw_ns")
        if g and f and g > f * 1.05:
            lag.append((g / f, f"2-D {shp}"))
    lag.sort(reverse=True)
    w("### Rows behind FFTW, worst first\n")
    if not lag:
        w("None: every row is at or above parity.")
    for r, name in lag:
        w(f"- {name}: {r:.2f}×")
    w("\nWhy each band trails, and the levers already measured against it "
      "(kept or dropped), are in the dated rounds of the repository's "
      "BENCHMARKS.md, not regenerated here.")

if __name__ == "__main__":
    main()
