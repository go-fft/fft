# Round 28: Cascade Lake, 64 to 32768 points

Measurements behind BENCHMARKS.md's Round 28. Host: Intel Xeon (Cascade
Lake), cfarm151, an 8-vCPU VM with AVX-512, FFTW 3.3.10 built with its AVX-512
codelets. Every run is one core pinned with `taskset -c 3`, `GOMAXPROCS=1`,
cross-compiled test binaries (go1.27.1); each file logs the load average at
its start and end (host clock, UTC).

- `raw/sweep1.txt`: `TestIntelSweep -intel.sweep=3,20,6,14`, every ordering of
  radix-4/8/16 passes from 64 to 16384 points in modes `w` (main's widths),
  `a` (AVX2 only), `s` (Round 23's split, AVX2) and `sw` (split, final pass
  AVX-512), three rotated rounds; `scripts/sum.py` summarizes.
- `raw/sweep2.txt`, `raw/sweep3.txt`: the same in modes `w` and `z` (the
  512-bit split layout), 256 to 16384; sweep2 with the first build (Z16..Z19
  and Z30, Z31 written), sweep3 with the final kernels.
- `raw/ab1.txt` (`scripts/ab1.sh`, seven rounds; 64 and 128 with the 512-bit
  layout allowed below 256), `raw/ab2.txt` (`scripts/ab2.sh`, fifteen rounds):
  `TestIntelAB`, the finalists per length in one process, order rotated.
- `raw/col5.txt`: 2-D 512² and 1024², the branch against a build whose N-D
  strips keep main's factorization (`scripts/ab.sh`, five rounds).
- `raw/e2e5-zmm16.txt`, `raw/e2e15-zmm16.txt`, `raw/e2e7-zmm16.txt`:
  `BenchmarkAB`, main against the first build, five, fifteen and seven rounds;
  `raw/comp7.txt`: four composites alone, the same two binaries;
  `raw/e2e7-zmm16-zeroed.txt`: main against a build zeroing Z16..Z31 on exit;
  `raw/e2e7-spill.txt`: main against the final kernels; `raw/zero9.txt`,
  `raw/spill9.txt`: the final kernels against the zeroing and the first build
  on the powers of two. `scripts/aban.py` summarizes any of them.
- `raw/e2e15-v018.txt`, `raw/e2e15-final.txt`: main (v0.18.0, v0.17.0) against
  the pull request rebased on it, fifteen rounds.
- `raw/d1024.txt`: 2-D 1024² alone, main against the branch, eleven rounds at
  1 s; `raw/d1024b.txt`: the branch against a build without 1024 in
  `intelSplitTable512`; `raw/d1024c.txt`: main against a build with the 512-bit
  layout off (tables kept); nine rounds each.
- `raw/mutants.txt`: thirteen mutations (`scripts/mutate.py`, `scripts/muts.sh`),
  each against the per-pass test, the transform test and three other suites.
- `parity-main1/`, `parity-br1/`, `parity-br2/`, `parity-main2/`: the parity
  harness (`scripts/parity.sh`, `scripts/run-remote-pinned.sh`), main
  (v0.16.1) and the branch before its rebase on v0.17.0 (whose changes are
  arm64 code and a scalar-pass wrapper), back to back on one pinned core.
