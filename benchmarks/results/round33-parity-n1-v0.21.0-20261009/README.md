# Round 33 raw data (2026-10-09)

Neoverse-N1 parity re-measured at v0.21.0 (bb351ec), measurement only, no
library change. Host cfarm424 (64 cores, used by this round alone), one pinned
core (`GOMAXPROCS=1 taskset -c 40`), the same harness, rows and method as
Round 27's `parity-br`/`parity-br2`: `benchmarks/remote/run.sh` with
Round 24's `-1` suffix (`scripts/run-remote-pinned.sh`), FFTW 3.3.10 built
from source with NEON, numpy 2.5.3 / scipy 1.18.1 in the host's venv. Binaries
cross-compiled with go1.27.1 (`GOOS=linux GOARCH=arm64`, `-ldflags='-s -w'`).
gonum is v0.17.0 here (Round 27: v0.16.0).

- `neoverse-n1/parity-a/`, `neoverse-n1/parity-b/`: two full runs back to
  back (`scripts/parity.sh a b`), each correct 24/24; `REPORT.md` per run
  with the raw `go_bench.txt`, `go_plan.txt`, `fftw.json`, `ref.json`.
- `neoverse-n1/parity-runs.log`: one-minute load at the start and end of each
  run (host clock, UTC). Each run is gated on a load below 1.5; the ~1.0 at the
  end is the run's own pinned core.
- `REPORT.md`: the mean of the two runs per row, and go-fft's time against
  v0.17.0 (mean of Round 27's `parity-br` and `parity-br2`).
