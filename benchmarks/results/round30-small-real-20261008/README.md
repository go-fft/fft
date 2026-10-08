# Round 30 — the smallest sizes: RFFT 256 and complex 256 (2026-10-07/08)

The measurements behind Round 30 of BENCHMARKS.md. Hosts: Zen 3 (cfarm420,
AMD EPYC 7773X) and Neoverse-N1 (cfarm424), each used by this round alone,
one pinned core (`GOMAXPROCS=1 taskset -c 40`); the 1-minute load is printed
at the head of every file (cfarm420 2.0–4.9 from other users, cfarm424
0.0–1.3).

| file | what |
|:--|:--|
| `zen3/breakdown-main.txt` | main: RFFT/IRFFT split into half transform, untangle, pools (`TestR30Real`); the untangle kernel with 1, 2, 3 interleaved steps (`STEPS`, exploratory build); `TestR26Overhead` |
| `zen3/passes-orders-256.txt` | main: complex 256's passes, and every radix order of 256 |
| `zen3/direct.txt`, `n1/direct-orders.txt` | the RealPlan buffer kept (pool) or dropped (direct), same binary, rotated; N1: radix orders of 128 and 256 |
| `zen3/steps-dup.txt`, `zen3/steps-halves.txt` | the untangle kernels: 1 step (Round 4), 3 steps, 2 and 3 steps with VMOVDDUP twiddles (s22, s23), the mirror as two 128-bit halves (s33) |
| `n1/pow2-sweep.txt` | every ordering of radix-2/4/8 passes for 2^5 … 2^13 against the rule (`TestR30Pow2Sweep`) |
| `*/ab-main-branch-15*.txt` | `BenchmarkR30AB`, main and branch binaries alternating, fifteen rounds (two runs on Zen 3) |
| `zen3/c64-alone-15.txt`, `layout-*.txt`, `bisect-11.txt`, `offsets-*.txt`, `overhead-main-branch.txt` | the complex-64 row on Zen 3 (see BENCHMARKS.md) |
| `*/breakdown-branch.txt`, `n1/breakdown-main-branch.txt` | the RFFT split again, main and branch |
| `zen3/mutants.txt` | fourteen generator mutations, each built and run on Zen 3 |
| `*/suites.txt` | both packages' suites on each host, branch binaries |
| `*/parity/` | four parity runs per host (main, branch, branch, main), `parity-ratios.txt` |

Scripts: `ab.sh` (suites, then main/branch alternating), `abn.sh` (n
binaries, the order rotated every round), `aban.py` (medians and per-round
ratios), `mutate.py`, `parity30.sh`, `fixreport.sh` (adds the `-1` suffix a
pinned run's benchmark names lack, which `report.py` needs, and rebuilds
`REPORT.md`), `parity.py` (go-fft ÷ the median of FFTW's four runs).
