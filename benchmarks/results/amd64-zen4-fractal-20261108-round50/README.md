# Round 50: is the 1,024 gap an L1D working-set effect? (Zen 4, 2026-11-08)

Host `fractal` (Ryzen 9 7950X, bare metal), method of Round 49
(`../amd64-zen4-fractal-20261108/README.md`): one pinned core with an idle SMT
sibling, `GOMAXPROCS=1`, counters per transform from `perf stat` at 200,000 and
1,000,000 iterations (difference over difference), user mode, one process at a
time, load 0.1–0.4 at every start. go-fft is v0.28.12 (the AVX2 route at 256,
512 and 1,024; at 2,048 the AVX-512 route, which Round 49 kept). FFTW is the
3.3.10 AVX-512 build, plans by `FFTW_MEASURE`. `pmu stat` gained
`-benchname BenchmarkPow2_GoFFT` to reach 512 and 2,048.

Hypothesis: at 1,024 go-fft's working set exceeds the 32 KiB L1D and FFTW's
does not, so its extra L1D fills from L2 are the remaining 1.31× gap.

## Counters per transform

| complex | | cycles | instructions | IPC | L1D fills from L2 (all) | of which demand | go ÷ FFTW cycles | go ÷ FFTW instructions |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 256 | go-fft | 938 | 4,049 | 4.32 | 0.02 | 0.17 | 1.31 | 1.72 |
| | FFTW | 714 | 2,348 | 3.29 | 0.02 | 0.03 | | |
| 512 | go-fft | 2,124 | 8,339 | 3.93 | 158 | 68 | 1.20 | 1.49 |
| | FFTW | 1,767 | 5,592 | 3.16 | 245 | 162 | | |
| 1,024 | go-fft | 5,255 | 18,413 | 3.50 | 1,582 | 662 | 1.33 | 1.49 |
| | FFTW | 3,943 | 12,338 | 3.13 | 913 | 449 | | |
| 2,048 | go-fft (AVX-512 route) | 14,496 | 27,738 | 1.91 | 5,529 | 2,634 | 1.40 | 1.02 |
| | FFTW | 10,367 | 27,235 | 2.63 | 2,827 | 1,593 | | |

(`ls_any_fills_from_sys.local_l2`, `ls_dmnd_fills_from_sys.local_l2`; the
`.all` event reads the same; hardware-prefetch fills are the difference. Raw
output: `stat-*.txt`. Counts are 64-byte lines.)

## Bytes touched per transform

go-fft, from the plan (`skPlan`: passes, `twX` tables, scratch):

| complex | passes | twiddle tables | src + dst + scratch | total |
|--:|:--|--:|--:|--:|
| 256 | 8·8·4 (frame kernel `ZnPlan`, buffers on the stack) | 3,968 B | 4 + 4 + 4 KiB | about 16 KiB |
| 512 | 8·8·8 (frame kernel) | 8,064 B | 8 + 8 + 8 KiB | about 32 KiB |
| 1,024 | 8·8·16 (three passes, `sync.Pool` scratch) | 16,128 B | 16 + 16 + 16 KiB | 63.8 KiB |
| 2,048 | 4·8·8·8 (AVX-512) | 32,640 B | 32 + 32 + 32 KiB | 127.9 KiB |

FFTW 3.3.10, from `fftw_print_plan` and `dft/simd/common/t1fv_16.c` (30
doubles of twiddle per column, so (r-1)·m·16 B): 1,024 is `n2fv_64` (out of
place, constants only) then `t1fv_16_avx512` in place on the output: 16 + 16
KiB + 15,360 B = 47 KiB. 2,048 (`n2fv_64` + `t1fv_32_avx512`): 64 KiB +
31,744 B = 95 KiB. 256 (`n2fv_8` + `t2fv_32`) and 512 (`n2fv_16` + `t2fv_32`)
use the compact `t2` twiddles, not opened here: at most 12 and 24 KiB.

## Reading

- The fills do scale with the working set past 32 KiB, in both libraries:
  go-fft 0.02, 158, 1,582 at 256, 512, 1,024 (about 16, 32, 64 KiB); FFTW
  0.02, 245, 913 (12, 24, 47 KiB at most). At 1,024 go-fft fills 1.73× as many
  lines (101 KB, 1.6× its working set; FFTW 58 KB, 1.2×). By the task's own
  criterion the hypothesis is not refuted.
- It does not explain the gap. The cycle gap is 1.31, 1.20, 1.33 at 256, 512,
  1,024: it is there at 256, where go-fft takes no fill, and at 512, where it
  takes fewer than FFTW. At those sizes go-fft's IPC (4.32, 3.93) is the
  higher one; what differs is the instruction count, 1.72× and 1.49× FFTW's
  (go-fft's AVX2 passes against FFTW's AVX-512 codelets, which Zen 4 runs as
  two 256-bit halves). At 1,024 the count is 1.49× again; go-fft's IPC (3.50)
  is still above the AVX-512 plan's (3.13, fewer and wider instructions) and
  below FFTW's AVX2 build's (3.93, Round 49).
- An upper bound for the fills' share at 1,024: go-fft's cycles per
  instruction rise 0.232 (256) to 0.285 (1,024), 987 cycles over the 18,413
  instructions, while its fills rise by 1,582; FFTW's rise 0.304 to 0.320, 193
  cycles over 913 fills. Charging all of go-fft's rise to the fills, and none
  to the radix-16 pass's different instruction mix, 987 of the 1,312 cycles is
  the most they can be (75%); at FFTW's cost per fill (0.21 cycles) the 669
  extra fills are 140 cycles (11%).

## Cheapest reduction: not available without changing operation order

- Last pass into dst, no scratch round trip: already so. Three passes write
  dst, scratch, dst (`skPlan.transform`).
- No scratch at all needs an in-place pass. Stockham passes are out of place;
  FFTW's second pass (`t1fv_16`) is a decimation-in-time twiddle codelet that
  works in place. That is another algorithm, with another operation order,
  not the bit-identical layout change asked for.
- Smaller or shared twiddle tables: the radix-8 first pass reads three
  streams of W^(j·i) (j = 1..3 would need W^2, W^3 from W: extra multiplies,
  other bits).
- Blocking the later passes in L1 (the cascade of Round 20 with a 512-point
  group): `kernels.StockhamLastRun` has no radix-16 last pass, so the plan
  8·8·16 panics (`index out of range [16] with length 9`) when forced; a
  radix-16 `cascadeLast` kernel would be a go-asmgen generator change, for a
  schedule that Round 35 found a tie with the AVX2 kernels at 65,536.
- The frame kernel (`ZnPlan`) at 1,024: its frame is two buffers (16,504 B
  at 512); at 1,024 that is 32 KiB of stack plus src, dst and 16 KiB of
  twiddles, more bytes than today's 64 KiB, so it cannot shrink the working
  set.

## Verdict

Not refuted by the scaling test, refuted as the explanation of the gap:
256 and 512 show 1.31× and 1.20× with no or fewer fills; the instruction count
is 1.5–1.7× FFTW's at every size; at 1,024 the fills can account for at most
about a third to three quarters of the cycles, and the only designs that cut
them change the operation order or need a new kernel. No code change, no
release; the default stays bit-identical. Not run: cfarm420, cfarm151 and
cfarm424, since nothing changed for them.
