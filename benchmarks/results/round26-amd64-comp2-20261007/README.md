# Round 26: composites and small real sizes on AMD Zen 3

Measurements behind BENCHMARKS.md's Round 26. Host: AMD EPYC 7773X (Zen 3),
cfarm420, a 128-thread VM; every timing on one core pinned with `taskset -c
40`, `GOMAXPROCS=1`, cross-compiled test binaries (go1.27.1). The load average
is logged in each file (2.2–3.8 on 2026-10-06 evening, 1.2–2.1 on 2026-10-07).
No Intel host was used: the Intel routing is unchanged.

`zen3/`:
- `loc1.txt`: main (v0.16.1) plus the Round 26 test file. `TestR26Passes`
  (each pass alone, on the buffers and in the layout the transform gives it,
  rotated with the whole transform, seven rounds), `TestR26Real` (RFFT whole,
  its half-length complex transform alone, its untangle alone, nine rounds)
  and `TestR26AB` (factorizations of each length against the current one,
  nine rounds, the order rotated every round). `s0`..`s4` in a pass name is
  its split-layout mode.
- `rule1.txt`: `TestR26Rule`, Round 24's Intel composite rule (radix 12, else
  a radix-16 tail) against the current AMD factorization, for every length it
  changes up to 16384, seven rounds.
- `pfa1.txt`: the first build with the radix-10/15/20 kernels, before any
  routing change: `TestR26AB` (nine rounds) and `TestR26Passes` (five) on
  factorizations with those passes.
- `sweep1.txt`: `TestR26Sweep`, every factorization `comp2Candidates` builds
  for each of the 148 lengths 2^e·3^a·5^b (e >= 1, a+b >= 1) up to 16384,
  five rounds: `factorization=median ns/current÷this`. The current
  factorization comes first. `scripts/an_sweep.py sweep1.txt
  scripts/rule_r6.py` scores the chosen rule against it; `scripts/model.py`
  is the linear pass-cost fit that was tried and not used.
- `ab-main-branch-5.txt`, `ab-main-branch-15.txt`: `BenchmarkR26AB`, main
  (`abA.test`) against the branch (`abB.test`), interleaved, the order
  alternating every round (`scripts/ab.sh`, Round 24's), five and fifteen
  rounds; `scripts/aban.py FILE abA.test abB.test` summarizes.
- `real-overhead.txt`: `TestR26Real` and `TestR26Overhead` (Plan.FFT, the
  plan's transform, its passes on a fixed buffer, a sync.Pool Get and Put
  alone), main and branch, eleven rounds.
- `half-scratch.txt`, `ab-rfft-halfscratch-11.txt`: the dropped RFFT change
  (the half transform on the RealPlan's buffer; `scripts/half-scratch-dropped.patch`).
- `scratch-cost.txt`: `TestR26ScratchCost`, a sync.Pool Get and Put against
  a one-slot atomic cache.
- `mutants.txt`: eight mutations of the radix-10/15/20 generator
  (`scripts/mutate.py`), each run against `TestComp2EachPassMatchesScalar`
  and `TestComp2TransformMatchesScalar`.
- `parity-*`: the parity harness (`benchmarks/remote`, pinned,
  `scripts/parity26.sh`) for main, the branch, the branch again and main
  again, back to back.
