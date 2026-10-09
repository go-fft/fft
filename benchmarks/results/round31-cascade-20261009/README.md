# Round 31 — Cascade Lake (cfarm151), 2026-10-08/09

Measurements behind BENCHMARKS.md's Round 31. One pinned core
(`GOMAXPROCS=1 taskset -c 3`); every file gives the 1-minute load at its start
and end.

- `raw/rows*.txt` — `TestR31Rows2D`: the column and row steps of an m×n
  PlanN with the row plan as the variant (`rows2s.txt`, `rows1024h.txt`:
  eight rows repeated, so from cache; `rowsm*.txt`: m×1024).
- `raw/ab1024.txt` — `TestIntelAB`, 1-D 1024 out of place, three orders.
- `raw/whole1.txt` — `TestR31Whole`: whole N-D transforms, row order off and
  on, one src and dst shared, one process.
- `raw/ab-r31-9.txt`, `raw/ab-v018-11.txt` — first change only, between
  binaries (main, v0.18.0).
- `raw/comp1.txt`, `raw/comp2.txt` — `TestR31Comp`: composite rules,
  interleaved and with the 256-bit split layout.
- `raw/real1.txt` — `TestR30Real` on main's RFFT split.
- `raw/unt1.txt`, `unt2.txt`, `unt3.txt` — `TestR31Untangle`: 512-bit
  untangle off and on (`unt1` before `clUntangleWide`).
- `raw/ab-r31-e2e.txt` — `BenchmarkR31AB`, main (`bb351ec`) against the
  branch, eleven rotated rounds.
- `raw/accuracy.txt` — `scripts/acc31.py` on `TestR31Export`'s bytes.
- `raw/mutants.txt` — generator and guard mutations.
- `parity/` — the parity harness, main, branch, branch, main;
  `parity/ratios.txt` from `scripts/partool.go.txt`.

`scripts/g.sh` and `scripts/ab.sh` gate on the load and rotate the order;
`scripts/*.go.txt` are the small summary programs (rename to `.go` and `go
run`). The first parity attempt ran the stale binaries in
`~/gofft-bench/runpkg` because `benchmarks/remote/run.sh` changes to that
fixed directory; it was discarded and rerun with Round 28's
`run-remote-pinned.sh`, which runs the package it sits in.
