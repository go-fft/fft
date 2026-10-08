#!/usr/bin/env python3
# parity.py MAIN1 MAIN2 BR1 BR2: go-fft time (mean of a variant's two runs) ÷ median FFTW time of all four
# runs, per REPORT.md row, and FFTW's own range.
import re, sys, statistics as st
def rows(path):
    out, sec = {}, ""
    for line in open(path + "/REPORT.md"):
        if line.startswith("## "): sec = line[3:].split("(")[0].strip()
        m = re.match(r"^\| ([\dx,]+)[^|]*\| ([\d,]+) \([^)]*\) \| ([\d,]+) \(", line)
        if m: out[(sec, m.group(1))] = (float(m.group(2).replace(",", "")), float(m.group(3).replace(",", "")))
    return out
m1, m2, b1, b2 = (rows(p) for p in sys.argv[1:5])
print(f"{'section':34} {'N':>10} {'FFTW med':>10} {'FFTW range':>17} {'main':>6} {'branch':>6}")
for k in m1:
    if not all(k in r for r in (m2, b1, b2)): continue
    f = [r[k][1] for r in (m1, m2, b1, b2)]
    fm = st.median(f)
    gm = (m1[k][0] + m2[k][0]) / 2; gb = (b1[k][0] + b2[k][0]) / 2
    print(f"{k[0][:34]:34} {k[1]:>10} {fm:10.0f} {min(f):8.0f}-{max(f):<8.0f} {gm/fm:6.2f} {gb/fm:6.2f}")
