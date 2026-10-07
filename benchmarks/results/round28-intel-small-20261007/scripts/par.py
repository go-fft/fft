#!/usr/bin/env python3
"""par.py DIR...: go-fft ns / FFTW ns and their ratio per row of each parity REPORT.md."""
import re,sys
def rows(f):
    sec=None; out={}
    for l in open(f):
        if l.startswith('## '): sec=l[3:].split('(')[0].strip()
        c=[x.strip() for x in l.split('|')[1:-1]]
        if len(c)>3 and sec and c[0][:1].isdigit():
            num=lambda s:int(s.split()[0].replace(',',''))
            if sec.startswith('Plan'): continue
            g,f=(num(c[1]),num(c[3])) if sec.startswith('2-D') else (num(c[1]),num(c[2]))
            out[(sec[:22],c[0].split()[0])]=(g,f)
    return out
R=[rows(d+'/REPORT.md') for d in sys.argv[1:]]
for k in R[0]:
    print(f"{k[0]:22s} {k[1]:>10}", ' '.join(f"{r[k][0]:>9}/{r[k][1]:<9} {r[k][0]/r[k][1]:4.2f}" for r in R))
