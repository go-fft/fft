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
