# Round 17 raw data (2026-10-05)

The measurements behind BENCHMARKS.md's Round 17: small and mid sizes on amd64.

- `zen3-parity/`, `cascadelake-parity/`: `benchmarks/remote/run.sh` on cfarm420
  (AMD EPYC 7773X, Zen 3) and cfarm151 (Cascade Lake), pinned to one core with
  `taskset`, so the 2-D rows there are single-core, against single-threaded
  FFTW. The reports were regenerated locally with `report.py`, after adding the
  `-1` GOMAXPROCS suffix the parser expects (a pinned run prints none).
  `go_bench_2d_allcores.txt` is the same binary's 2-D rows on all cores.
  The Zen 3 package was built at the commit before the inlined out-of-place
  path in `skPlan.transform`; the Cascade Lake one at the final code.
- `raw/`: interleaved runs, one line per benchmark and binary
  (`scripts/ab.sh`; `GOMAXPROCS=1 taskset -c K` unless the file name ends in
  `-mc`). `ROUND` lines carry the host's load average. File suffix = host
  (420 Zen 3, 151 Cascade Lake, 13 Haswell).
  - `decomp-*`: where the time goes (`BenchmarkDecomp`, `BenchmarkDecomp2D`) on `main`.
  - `sw1-*`, `sw2-*`: strip widths and every radix ordering (`BenchmarkOrders`).
    `sw1-13` ran at load 25 on 24 threads and is not used.
  - `ab-dup-*`: pre-duplicated twiddle tables (`dup.test`) against the strips
    build without them (`cur.test`).
  - `ab-1-*`: `main.test` (630f2e4) against `wid.test` (strips, widths 16/32,
    pocketfft order) and `ord.test` (plus odd radices first).
  - `fan-*`: `BenchmarkFanout`, all cores.
  - `ab-fin*`: `main.test` against `fin.test` (all changes but the inlined
    transform path) and `fin2.test` (final); `ab-fin15-*` is the fifteen-round
    rerun of the rows within their spread; `ab-inl-151` compares `fin.test` and
    `inl.test` (inlined path; its batched kernels carried a deliberate mutation
    left from a mutation check, which no row of that file runs).
- `scripts/`: `ab.sh` (the interleaving), `ab3.py` (medians, ratios, spreads),
  `med.py`, `sw2.py`, `rules.py` (scores radix-order rules on `sw2-*`).
