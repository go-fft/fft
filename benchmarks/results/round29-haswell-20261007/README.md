# Round 29: Haswell, AVX2 without AVX-512

Measurements behind BENCHMARKS.md's Round 29. Host: Haswell (cfarm13, two
Intel Xeon E5-2620 v3, bare metal, `perf`). Every run is one core pinned with
`taskset -c 10`, `GOMAXPROCS=1`, cross-compiled test binaries (go1.27.1),
started by `scripts/gate.sh` only once the 1-minute load was below 2, and
rerun when it had reached 3 by its end; each chunk logs the load at its start
and end (host clock, CEST). The host has no FFTW.

- `raw/sweep-pow2.txt`: `TestIntelSweep -intel.sweep=3,20,e,e`, e = 6..16,
  every ordering of radix-4/8/16 passes in modes `w`/`a` (interleaved, the
  same on this CPU) and `s` (split), three rotated rounds (`scripts/phase1.sh`;
  `scripts/swan.py` summarizes; the 4096 chunk was discarded once and rerun).
- `raw/comp1.txt`: `TestHswComp` (first form: main's factorization and
  `comp2Factors'`, interleaved and split), five rotated rounds;
  `raw/sweep-comp.txt`: `TestR26Sweep`, every `comp2Candidates`
  factorization of the 145 composites, five rotated rounds;
  `scripts/an_sweep.py` with `scripts/rule_c2.py` (Round 26's rule) or
  `scripts/rule_hsw.py` (this round's) scores a rule against it.
- `raw/ab-pow2.txt` (`scripts/phase2.sh`): `TestIntelAB`, the finalists per
  power of two from 32 to 2^20, seven rotated rounds.
- `raw/comp3.txt`, `raw/ab-s1.txt` (`scripts/phase3.sh`, binary `r29b`):
  `TestHswComp` in its final form (Round 24's rule, `comp2Factors`,
  `hswComp2Factors`; interleaved, split, and `s1`, split runs of two passes
  or more only, from a knob since removed), and the powers of two in modes
  `a`, `s`, `s1`.
- `raw/pmu.txt` (`scripts/phase4.sh`): `perf stat`, two event groups per
  variant (ports 0, 1, 5 and resource stalls; L1D replacements, 4K
  aliasing, store-buffer stalls, L1D-pending stalls).
- `raw/tests-haswell.txt`: the full suites of both packages on this host with
  the branch's routing, and `TestHswExport`'s list; `raw/accuracy.txt`:
  `scripts/acc.py`, numpy.fft on the exported input bytes against both
  factorizations' outputs.
- `raw/e2e5.txt`, `raw/e2e15.txt` (`scripts/e2e.sh`): `BenchmarkAB`, main
  (v0.19.1) and the branch alternating every round, five and fifteen rounds;
  `scripts/aban.py FILE main.test br.test` summarizes. In `e2e5.txt` the
  branch's third round ran its first six rows at half speed (a burst of
  another user's load between two load checks); the medians absorb it.
