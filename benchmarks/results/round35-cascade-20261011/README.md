# Round 35 — Cascade Lake (cfarm151), 2026-10-11

These are the measurements behind BENCHMARKS.md's Round 35. They ran on one pinned core
(`GOMAXPROCS=1 taskset -c 3`), and every timed file gives the 1-minute load at its
start and end (`scripts/g.sh` waits for a load below 1.5).

- `raw/loc1.txt` — `TestCsCascadeParts`, `TestR26Overhead`, `TestR26Passes`,
  `TestR26Real`: the time located part by part.
- `raw/sweep1.txt`, `raw/ord1.txt`–`ord3.txt` — every radix-4/8 order at
  2^16 … 2^20, the best against the rule, the blocked schedule at 16384 and
  32768.
- `raw/parts2.txt`, `raw/grp1.txt`, `raw/pair1.txt` — the blocked schedule's
  parts, the 512-bit split layout in its groups, the pair's chunk width.
- `raw/e2e1.txt`–`e2e3.txt` — `TestCsCascadeE2E`, main's table against this
  one through the exported API.
- `raw/r9a.txt`, `raw/comp9a.txt`, `raw/comp9b.txt` — radix-9 factorizations
  and the rule over 73 lengths.
- `raw/zn1.txt`, `raw/zn2.txt`, `raw/nd1.txt` — frame against pool, before
  and after the 512-bit kernels joined; N-D plans.
- `raw/unt1.txt` — the 512-bit untangle's bound.
- `raw/tj1.txt`–`tj4.txt` — the untangle tail jump against two calls
  (`tj4` is the one quoted).
- `raw/rows512.txt`, `raw/place512*.txt` — 2-D 512² by axis and by
  allocation.
- `raw/fftw-plans.txt` — FFTW's plans and times (`scripts/csfftwplan.c`).
- `raw/accuracy.txt`, `raw/export.txt` — `scripts/acc35.py` (numpy 2.5.3) on
  `TestCsExport`'s bytes, with a planted error.
- `raw/mutants.txt` — the mutations, by `~/.local/bin/mutate` and
  `scripts/mrun.sh`.
- `parity/` — the parity harness, main, branch, branch, main (`p-*`), the
  loads (`parity-loads.txt`), and `ratios-main.txt` / `ratios-branch.txt`
  from `scripts/partab.go.txt` (rename to `.go`).

`scripts/cov.sh` builds both packages' `-coverpkg` test binaries, runs them
on the host and merges the profiles.
