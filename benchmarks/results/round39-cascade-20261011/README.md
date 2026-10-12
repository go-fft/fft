# Round 39 — Cascade Lake (cfarm151), 2026-10-11

The measurements behind BENCHMARKS.md's Round 39. Every timed run used one pinned
core (`GOMAXPROCS=1 taskset -c 3`) and waited for a 1-minute load below 1.5
(`scripts/g.sh`, `scripts/ab.sh`). Each file gives the load at its start and end.
"Main" is v0.26.1 (`365a1e3`) throughout.

- `raw/pages1.txt`, `raw/pages2.txt`, `raw/procs1.txt`–`procs6.txt`: `TestCl3Pages`,
  2-D 512² on Go-heap buffer pairs and on buffers forced to 4 KB or 2 MB pages,
  whole and per axis, with how much of each Go-heap buffer the kernel held on
  huge pages. `pages1` is the first process of the session. `pages2` churned the
  heap first (`-cl3.scav`). `procs*` are six later processes.
- `raw/scr1.txt`, `raw/scr2.txt`: `TestCl3Scratch`, the column step with its strip
  scratch at eleven placements.
- `raw/mem1.txt`–`mem3.txt`: `TestCl3Mem`, the columns and rows beside memory probes
  (latency, bandwidth, and a read of the columns' own bytes).
- `raw/sp1.txt`: `TestCl3StripPasses`, the column step pass by pass.
- `raw/sv1.txt`–`sv5.txt`: `TestCl3StripVars`, strip widths (and copy-in/copy-out
  variants) on 2-D and 3-D shapes. `sv2-*` end in a panic of the harness on its
  last shape (3-D, an axis without strips); every row before it is complete.
- `raw/fr1.txt`: `TestCl3FrameRowsAB`, the N-D rows on and off the frame kernels.
- `raw/ab1.txt`: `TestZmPlace` (Round 36's placement harness), main against the
  branch, the binaries alternated for nine rounds, four placements each
  (`scripts/abtab.go.txt` tabulates it).
- `raw/loc1.txt`, `raw/ord1.txt`: complex 1080, 1920 and 65536 per pass, RFFT 65536
  and 2^20 split. Then 21 factorizations of 1080 and 21 of 1920 against the rule.
- `raw/accuracy.txt`: `scripts/acc39.py` (numpy 2.5.3) on `TestCl3Export`'s bytes,
  with a planted error.
- `raw/mutants.txt`: 14 mutations (`scripts/mutgen.go.txt`, `scripts/mrun.sh`).
- `raw/coverage.txt`: both packages' coverage binaries on cfarm151, merged.
- `parity/`: the parity harness (`scripts/mkpkg.sh`), main, branch, branch, main,
  with `parity-loads.txt` and the ratios by
  `../current-numbers-v0.26.0/partab.go.txt`.
