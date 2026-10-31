# Round 43 — Zen 3, complex 1024 against FFTW (2026-10-31): a negative result

Host: cfarm420 (AMD EPYC 7773X, Zen 3, KVM, no PMU), one pinned core
(`GOMAXPROCS=1 taskset -c 40`), load 6.1–7.0 of 128 cores (each log gives
it). Binaries cross-compiled with go1.27.1 from v0.28.6 plus this round's
test file. Nothing here is merged: no change beat v0.28.6.

## What FFTW 3.3.10 runs (`scripts/plan1024.c`, `scripts/fftwplace.c`)

FFTW_MEASURE and FFTW_PATIENT, out of place, pick one of two two-pass plans
from run to run, both within FFTW's own noise (1535–1616 ns best of 31):

- `(dft-ct-dit/32 (dftw-direct-32 "t2fv_32_avx2") (dft-direct-32-x32 "n2fv_32_avx"))`
- `(dft-ct-dit/16 (dftw-direct-16 "t2fv_16_avx2") (dft-direct-64-x16 "n2fv_64_avx2"))`

Read in FFTW 3.3.10's `dft/simd/common/{n2fv_32,t2fv_32}.c` and
`simd-support/simd-avx2.h`: both codelets work on the interleaved layout,
two complex values per YMM register (LD = 128-bit load + VINSERTF128); the
no-twiddle pass reads strided input and stores pairs transposed (STN2); the
twiddled pass runs in place on the output with a full twiddle table (VTW, 31
per vector), each product VZMULJ = three shuffles, a multiply and a
VFMSUBADD. The avx2 twiddle codelet is genfft's `-fma` output: 59 additions,
2 multiplications and 158 fused multiply-adds per 32-point butterfly pair
against 217 additions and 104 multiplications without FMA.

## Where our 1024 goes (`zen3/loc1.txt`, `TestZ43Passes`, 21 rounds)

| plan | pass 1 | pass 2 | pass 3 | sum | whole |
|:--|--:|--:|--:|--:|--:|
| 8·8·16 (default) | r8 split-out 541 | r8 split-in 510 | r16 final 589 | 1639 | 1687 |
| 8·8·16 FMA | 514 | 485 | 590 | 1590 | 1672 |
| 32·32 FMA | r32 split 919 | r32 final 775 | | 1694 | 1754 |

Whole transforms (`TestZ43AB`): every order of 4, 8 and 16 timed lost to
8·8·16 (0.81–0.99). FFTW in the same session: 1560–1593 ns median, so
8·8·16 runs 1.07–1.09× FFTW's median (the parity report's best-of-6 FFTW
column gives 1.10–1.13). Each of our three passes costs about a third; the
two-pass plans we can build are slower because the radix-32 kernels are.

## Hypotheses tested and refuted

1. **Buffer placement** (`zen3/loc2.txt`, `TestZ43Place`): src at 0, 576,
   1088, 2048, 2112, 3136 bytes (mod 4 KB) from dst. Ours moves 1.5% (1707 at
   offset 0, 1683 at 1088); FFTW does not move (±0.5%). Not the gap.
2. **FADD-port bound** (`zen3/loc3.txt`, `TestZ43Bal`, the kernels of the
   three passes rewritten by `scripts/faddbal`): one addition in every 2, 3
   or 4 moved to the FMA pipes as fma(a, 1, ±b), which is bit-identical to
   a ± b (the test checks every output bit). Every variant was SLOWER, more so
   the more additions moved: whole transform 0.91, 0.95, 0.96× (twice, in one
   process, rotated rounds). The kernels are not add-port bound.

## Reading

The instruction count, not one port, separates us from FFTW: about 17 100
instructions per transform here (three passes of 32 groups of ~175–185)
against an estimated ~12 800 for FFTW's two passes. Ours run at the higher
IPC, which is why the gap is ~8% rather than ~33%. A shorter transform needs
fewer passes with kernels as tight as the radix-8/16 ones, which the
radix-32 kernels (Round 38, Round 42) are not: they keep the 32 values
between their stages in memory.
