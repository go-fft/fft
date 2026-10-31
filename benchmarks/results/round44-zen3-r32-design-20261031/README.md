# Round 44 — a two-pass 32×32 plan for complex 1024 on AMD AVX2: design on paper (2026-10-31)

Gate set before writing any generator: the plan must count at most ~14,500
instructions per transform on paper (target ~13,000), against ~17,100 for
v0.28.6's 8·8·16. **It does not pass. Nothing was implemented.**

## Sources opened

FFTW 3.3.10, `dft/simd/common/n2fv_32.c` and `t2fv_32.c` (genfft headers),
`simd-support/simd-avx2.h` (LD, ST, STN2, VZMULJ, BYTWJ). Per butterfly pair
(two 32-point transforms, one YMM register = two complex values):

| codelet | FP ops (genfft header) | memory accesses |
|:--|:--|--:|
| n2fv_32, no FMA | 186 add + 42 mul = 228 | 80 |
| t2fv_32, no FMA | 217 add + 104 mul = 321 | 64 |
| t2fv_32, FMA (what avx2 runs) | 119 add + 62 mul + 98 fma = 279 | 64 |

Correction to Round 43: its "~12,800 instructions for FFTW" was an
underestimate. With the same counting rules as below, FFTW's own 32×32 plan
comes to about 15,700 per transform. That is 17,100 / 15,700 = 1.09, which
matches the measured 1.07–1.09.

## Register and spill schedule (best case)

The layout is interleaved, two transforms per register. Inside each
codelet the butterfly splits 4×8: four radix-8 butterflies, then the
internal twiddles W32^(n2·k1), then eight radix-4 butterflies. Sixteen YMM
registers hold one radix-8 butterfly and its scratch, so of the 32 values
between the stages about 16–24 must be spilled. genfft reports 59–72 stack
variables for these codelets. The arithmetic is taken to be as small as
genfft's, which would also need a new Go reference DAG with the same
operations in the same order for bit-identity.

Per butterfly pair, instructions:

| item | final pass, ido = 1, no twiddles | twiddled pass, ido = 32 |
|:--|--:|--:|
| loads | 64 (MOVUPD + VINSERTF128, two blocks per register) | 32 |
| FP arithmetic (genfft counts) | 228 | 321 default, 279 FMA |
| ×i rotations (permute + sign xor, not in genfft's count; ~24) | 48 | 48 |
| twiddle swaps (pre-duplicated table, one shuffle each) | — | 31 |
| spills, stores + reloads | ~40 | ~40 |
| stores | 32 | 32 |
| loop | 8 | 8 |
| **total** | **~420** | **~512 default, ~470 FMA** |

There are 16 butterfly pairs per pass, so:

- **Default mode: 16 × (420 + 512) ≈ 14,900.**
- **FMA mode: ≈ 14,200.**

## Verdict

- Default mode, the bit-identical one that the target is about: ~14,900, above the 14,500 gate. It also assumes:
  - genfft-level arithmetic, which nobody has derived for our kernels;
  - a spill count the 16 registers may not allow.
  So the realistic figure is higher.
- FMA mode: ~14,200 passes on paper, but only if both assumptions above hold.
- The radix-32 kernels that exist today (Round 38, Round 42) lost to 8·8·16, and their arithmetic is not genfft's.

Stopped here, as the gate requires.

## Follow-up: the user asked to build it anyway — and the premise fails on the existing kernels

1. **Licensing.** FFTW's codelets are GPL-2.0-or-later, while go-fft and
   go-asmgen are BSD-3. A Go reference that copies n2fv_32's or t2fv_32's
   expression DAG would be a derivative work, so it must not be done. It is
   not needed either. The 4×8 split that go-fft already uses (bfly32FMA)
   costs 376 real additions and 88 multiplications per 32 points against
   FFTW n1_32's 372 and 84, so the arithmetic already matches genfft.
2. **Instruction counts of the generated kernels** (`scripts/asmcount`,
   steady loops of v0.28.6's .s files):

   | kernel | per iteration | per transform |
   |:--|--:|--:|
   | skSplit8ISFwd / SIFwd (8 points × 4 lanes) | 176 | 2 × 32 × 176 = 11 264 |
   | skLast16AVX2 (2 blocks) | 188 | 32 × 188 = 6 016 |
   | **8·8·16 total** | | **17 280** |
   | skSplit32IIFwdFMA (32 points × 4 lanes) | 1 002 | 8 × 1 002 = 8 016 |
   | skLast32AVX2FMA (2 blocks) | 496 | 16 × 496 = 7 936 |
   | **32·32 FMA total** | | **15 952** |

   The FMA 32·32 plan that go-fft already has is at FFTW's count
   (~15 700), yet it runs slower than 8·8·16. Round 43, quiet host:
   1754 against 1687 ns, and 1672 for FMA 8·8·16. Its twiddled pass retires
   about 8 instructions per ns where every other pass retires about 10.
   The deficit is IPC, not instruction count, and without a PMU this VM
   cannot say which stall it is.
3. **The frame variant** (`z44frame-load25.txt`, `TestZ44Frame`,
   load 25): keeping the radix-32 values between stages in the stack frame
   instead of the 512-byte-strided output rules out 4 KB aliasing. The pass
   got slower: 1128 against 1016 ns.
4. **Best case for a new interleaved twiddled radix-32 pass** (t2fv-like,
   FMA): about 557 instructions per pair, 8 900 per pass, about 890 ns at
   10 per ns. Plus the final pass (786) that is about 1 680–1 720 ns, which
   does not beat FMA 8·8·16 (1672). Default mode needs 31 more instructions
   per pair.

Checkpoint rule: the radix-32 kernels measure slower than 8·8·16 at its
best, so I stop. Nothing is shipped.
