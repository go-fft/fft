# Round 24 raw data (2026-10-06)

The measurements behind BENCHMARKS.md's Round 24: smooth composite lengths on
Cascade Lake (cfarm151, AVX2 and AVX-512 kernels, 8 vCPUs) and Neoverse-N1
(cfarm424, 64 cores). Every run is one pinned core (`GOMAXPROCS=1 taskset -c
5` on cfarm151, `-c 40` on cfarm424); each file logs the one-minute load
before every round. Zen 3 was not used (it belonged to another round).

Binaries, cross-compiled with go1.27.1 from:

- `main`: 3754264 (v0.13), `main3`: ba859c6 (v0.14.0, Round 23; its change
  is AMD-only, so on Cascade Lake it runs the same code as `main`);
- `br` / `br1`: the branch before radix 12 (NEON final passes for any block
  count, split layout and odd-first order on arm64, radix-16 tails on Intel);
- `br2`: the branch with radix 12, before the rebase on v0.14.0; `br3`: the
  same, rebased.

Files, per host directory:

- `decomp-*.txt`: `BenchmarkR24Decomp`, each factorization whole and each of
  its passes alone (default plans first, then `scripts/facts.*`), five rounds.
  `decomp-main` is main; on N1 `decomp-lastpass` adds the NEON final passes
  for an odd l1 and radix 8, `decomp-split` the split layout as well;
  `decomp-radix12` (Cascade Lake) the radix-12 kernels. `scripts/decan.py`
  prints medians and spreads.
- `tails.txt` (Cascade Lake): `BenchmarkR24Tails`, every power-of-two tail
  after the odd passes for 29 lengths, three rounds (`scripts/tailan.py`).
- `rule12.txt` (Cascade Lake): `BenchmarkR24Rule12`, the default factorization
  against the radix-12 rule for the 96 lengths 2^e·3^a·5^b <= 16384 with a >= 1
  and e >= 2, three rounds.
- `ab-*.txt`: `BenchmarkR24AB`, interleaved, the order alternating each round
  (`scripts/ab.sh`), five rounds; `ab15-main-br3.txt` the fifteen-round rerun of
  the Cascade Lake rows within their spread. `scripts/aban.py` prints medians,
  the ratio of medians and the range of per-round ratios.
- `parity-{main,main2,br,br2}/`: `benchmarks/remote/run.sh` pinned with
  `taskset` (`scripts/run-remote-pinned.sh`, which adds the `-1` suffix
  report.py expects), main and branch back to back, twice; each correct 24/24.
  On Cascade Lake the `br` run's FFTW times came out 3–33% slower than in the
  three other runs (1000: 3,141 against 2,711–2,780 ns), so its ratios are not
  used.
