# Round 36 — Zen 3, the rows FFTW still led (2026-10-11)

Host: cfarm420 (AMD EPYC 7773X, Zen 3), one pinned core (`GOMAXPROCS=1
taskset -c 40`), load 1.2–2.3 from other users (each file gives it). Every
binary was cross-compiled with go1.27.1 from the branch or from v0.23.1
(2770704, "main"), stripped, and checked by SHA-256 on the host.

`zen3/`:

| file | what |
|:--|:--|
| `fftw-plans.txt` | FFTW_MEASURE plans and best-of-9 times for the sizes in question (`scripts/fftwplan36.c`) |
| `loc1.txt` | main: each pass alone (`TestR26Passes`), RFFT split into half and untangle (`TestR26Real`), Bluestein and 2-D 64²/128² taken apart (`TestZmLocate`) |
| `ab-comp1.txt` | every order of 1000's and 1080's radices (`TestR26AB`) |
| `ab-pow2-conv.txt` | orders for 2048, 960, 6144, 10240 (`TestR26AB`); every 7-smooth Bluestein length for 10007 (`TestZmConv`) |
| `ab-br1.txt` | in-process old ÷ new: Bluestein products, 2-D rows on the frame (`TestZmAB`) |
| `tails1.txt`, `pow2rule1.txt`, `pow2-ab15.txt` | the e >= 11 tails (`TestZmTails`, `TestZmPow2Rule`, `TestR26AB` fifteen rounds and per pass) |
| `odd1.txt` | factorizations with radix 3 and 5 interleaved and split (`TestZmOddAB`) |
| `sweep1.txt` | every composite to 16384 over the candidates (`TestZmSweep`) |
| `conf1.txt`, `conf2.txt`, `conf3.txt` (+ `-summary`) | the sweep's winners confirmed, fifteen and 21 rounds (`TestR26AB`) |
| `compall1.txt` | every composite, v0.23.1's plan against the branch's (`TestZmCompAll`) |
| `ab-main-br-9.txt`, `ab-main-br2-9.txt` (+ `.ratios.txt`) | main against the branch between binaries (`TestZmPlace`, `scripts/abn.sh`, `scripts/aban.py`); the first before the radix-3/5 work, the second on the final code |
| `accuracy.txt` | numpy on the exact exported bytes (`TestZmExport`, `scripts/zm_accuracy.py`) |
| `mutants.txt` | each mutation, built and run on Zen 3 |
| `parity/` | the parity harness, main, branch, branch, main (`scripts/parity36.sh`, `scripts/pkgparity.sh`) |
