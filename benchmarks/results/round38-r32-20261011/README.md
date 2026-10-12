# Round 38 — a radix-32 pass on amd64; Zen 3's RFFT parity at v0.26.1 (2026-10-11)

See BENCHMARKS.md, "Round 38". Hosts: Zen 3 (cfarm420, core 40) and Haswell
(cfarm13, core 10), `GOMAXPROCS=1 taskset`, go1.27.1 cross-compiled from
darwin/arm64. Main is v0.26.1 (365a1e3).

| file | what |
|:--|:--|
| `zen3/ab1.txt` | `TestR32AB`, every radix-32 factorization against the current plan, nine rotated rounds, branch 75780bd (`r32br.test`, SHA-256 in the file), started by `scripts/r32calm.sh.txt` below load 2.5 |
| `zen3/ab-exp1.txt` | first Zen 3 A/B with the experiment binary (pre-duplicated twiddles, variants prefixed d; `scripts/r32exp.diff.txt`, dup half), seven rounds |
| `zen3/passes1.txt` | `TestR32Passes`: each pass alone, nine rotated rounds |
| `zen3/spill1.txt`, `spill1.sha` | the spill experiment (`TestR32ExpSpill`, `scripts/r32exp_amd64_test.go.txt`; kern 1 = the timing-only kernels without the stores between the stages, wrong results) |
| `zen3/suites.txt`, `haswell/suites.txt` | both packages' suites with coverage (`scripts/r32suite.sh.txt`), merged profiles |
| `zen3/accuracy.txt` | `TestR32Export` + `scripts/r32_accuracy.py` (numpy 2.5.3), and the planted-byte check |
| `zen3/parity/` | five parity runs of v0.26.1 (`scripts/pkgparity.sh.txt`, `scripts/parity38.sh.txt`, binaries checked against `main.sha`); `runs*.log` give the loads; runs 1–3 ended above 3.5 and were rerun; `partab-par45.txt` is the table (runs 4 and 5, FFTW over all five), `partab-all5.txt` all five |
| `haswell/ab-loaded1.txt` | the A/B on Haswell, ungated: load 15–23 from another user's jobs (the gated run never started) |
| `haswell/ab-exp-dup-loaded.txt` | the pre-duplicated-twiddle A/B on Haswell at load 9, copied from the terminal |
| `haswell/mutants.txt` | 21 mutations (`scripts/mutbuild.sh.txt`), all killed |
| `scripts/r32zen3queue.sh.txt` | the four queue scripts that ran the Zen 3 jobs one at a time |
