# Round 51: why the AVX-512 route stalls on Zen 4, and the route that does not (2026-11-08)

Host `fractal` (Ryzen 9 7950X, Zen 4, bare metal), core 2 pinned, `GOMAXPROCS=1`,
load 0.2-1 at every start; Cascade Lake on cfarm151 (load 0.00). Medians of
interleaved rounds; the driver is `BenchmarkRound51` and `GOFFT_FORCE512`
(`round51_bench_amd64_test.go`, test-only): `1` is v0.28.11's route (interleaved
AVX-512 passes, no split layout), `split` is Cascade Lake's (the 512-bit split
layout), unset is the build's default.

## IBS, forced AVX-512 route, complex 1,024

`sudo -n perf record -e ibs_op// -c 100000`, 248K samples, by symbol:
`skPass4AVX512` 43.8%, `skLast8AVX512` 31.4%, `skPass8AVX512` 20.0%.
`ibs-annotate-skLast8AVX512-forced-avx512-route.txt` is the annotation of the
final pass. The samples are flat: 0.5-2.3% on every instruction of the loop,
the 24 `VINSERTF32X4` (the gather of four blocks into one register) 1.0-1.4%
each, 33% of the function together, the 7 stores 1.1-2.3%. No single
instruction stalls: the cost is the count of operations on pipes FP1 and FP2.

uops.info (Zen 4 rows), all one uop on FP1/2 with a measured throughput of 1.00
cycle: `VINSERTF32X4 zmm, zmm, m128, imm8` (latency 1), `VINSERTF64X4 zmm, zmm,
m256, imm8`, `VSHUFF64X2 zmm, zmm, zmm, imm8` (latency 4), `VBROADCASTF32X4 zmm
{k}, m128`; a merge-masked `VMOVUPD zmm {k}, m512` is one uop on FP0-3, 1.00;
`VPERMILPD zmm, zmm, imm8` one uop on FP1/2/3, 0.67. (The AMD Software
Optimization Guide for family 19h and Agner Fog's tables were not readable
here; the uops.info rows are what was opened.) Candidates tested against that:

| change to the final pass (forced AVX-512 route, 1,024) | time |
|:--|--:|
| v0.28.12 kernel (1 load + 3 `VINSERTF32X4` per register) | 1,103 ns |
| masked 128-bit lane loads (K2-K5): 4 FP0-3 uops | 1,185 ns (worse) |
| 256-bit halves: 2 `VINSERTF32X4 ymm` + 1 `VINSERTF64X4` | 1,075 ns (-2.5%) |
| contiguous 512-bit loads, no gather, wrong results | 1,034 ns (-6%) |

`VSHUFF64X2` transposition (8 contiguous loads, 16 shuffles, 8 register moves)
was within 2% of the halves. So the gather is at most 6% of the transform on
that route: it is not why the route loses to the AVX2 one (1,103 against 967
ns). What the AVX2 route has that the AVX-512 one does not is the split layout:
no `VPERMILPD`, no sign-flip `VPXORQ` per rotation.

## The 512-bit split layout on Zen 4

`sweep-split512-vs-default-base-kernels.txt`: the v0.28.12 binary, `split`
against the default (Zen 3's route up to 65,536 except 2,048; the interleaved
AVX-512 route above), seven rounds, each mode twice per round. Default time ÷
split time: 256 0.940, 512 0.957, 1,024 1.005, 2,048 1.018, 4,096 0.921, 8,192
0.897, 16,384 0.917, 32,768 0.982, **65,536 1.065, 131,072 1.034, 262,144
1.026**, 524,288 0.995, 2^20 0.962. Shipped for 65,536, 131,072 and 262,144.

`sweep-final-pass-gather-in-split512-route.txt` (`sb` v0.28.12 kernels, `sw`
256-bit halves, `sn` no gather, wrong results): in the split route the final
pass is the interleaved kernel and its gather is 15% at 256, 14% at 512, 9% at
1,024, nothing from 2,048; the halves recover about half of it.
`cascade-lake-halves-kernel-refuted*.txt`: the halves kernel on Cascade Lake
(8 rounds, load 0) is 3-4.6% slower at 256-1,024, so it is not shipped and
go-asmgen is unchanged.

## Result

`sweep-v0.28.12-vs-new-vs-fftw-avx512build.txt` (`sb` v0.28.12, `f` this change,
`fftw` FFTW 3.3.10 AVX-512 build, the fastest on this CPU; seven rounds; load
0.3-0.9). go-fft ÷ FFTW: 256 1.26, 512 1.25, 1,024 1.30, 2,048 1.36 (all
unchanged), 4,096 1.05, 8,192 1.07, 16,384 1.03 (unchanged); 65,536 0.936 →
**0.881**, 131,072 0.917 → **0.895**, 262,144 0.859 → **0.836**. The small-size
gap (256, 1,024) is not closed.
