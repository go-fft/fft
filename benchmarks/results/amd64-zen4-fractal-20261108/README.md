# Round 49: AMD Zen 4 with a readable PMU (2026-11-08)

Host `fractal`: AMD Ryzen 9 7950X (Zen 4, 16 cores, AVX-512), bare metal, no
hypervisor, Ubuntu kernel 6.8, `perf` 6.8.12 (`perf_event_paranoid=2`, `sudo`
for IBS only). The first x86 host here where cycles, instructions and IBS can
be read. No Go and no numpy on it: the Go side is cross-compiled
(`GOOS=linux GOARCH=amd64 go test -c`) and copied; FFTW 3.3.10 is built from
source twice, with `--enable-sse2 --enable-avx --enable-avx2 --enable-avx512
--enable-fma` (plans 1,024 as `t1fv_32_avx512` + `n2fv_32_avx2`) and without
`--enable-avx512` (`t1fv_32_avx2` + `n2fv_32_avx2`), both as `fftw_one`
(`benchmarks/cbench/fftw_one.c`).

Method. Everything on core 2 (`taskset -c 2`, its SMT sibling 18 idle),
`GOMAXPROCS=1`, one process at a time, interleaved rounds, the driver
`benchmarks/remote/pmu` refusing to start above its load gate (1.5 for the
parity runs; the gate was raised to 6 for the later runs because a second
user's desktop, 92% idle in `top`, kept the one-minute average at 2–9 through
D-state I/O; the highest value at a start was 2.6). Counters are per
transform: `perf stat` of the same benchmark at two iteration counts, the
difference divided by the difference in iterations, so planning, warm-up and
process start-up cancel. User mode only. Baseline = a clean checkout of the
v0.28.11 tag.

| file | what |
|:--|:--|
| `parity-base-v0.28.11-vs-fftw-avx512build.txt` | v0.28.11 against FFTW (AVX-512 build), 7 rounds |
| `parity-new-and-base.txt` | this change and v0.28.11 against both FFTW builds |
| `counters-1024-*.txt`, `counters-65536-new-and-fftw.txt` | the counter tables |
| `ma64-by-symbol-base-1024.txt`, `annotate-skPass4AVX512-ma64.txt` | `ls_misal_loads.ma64` samples |
| `ibs-op-by-symbol-1024.txt` | IBS op samples by function, v0.28.11 and this change |
| `exp-*.txt` | the experiments that chose the route (an environment switch that hid AVX-512 from the binary, not part of the change) |
| `final-base-vs-new-default-and-fma.txt`, `fft2-and-fma-first-cut.txt` | this change against v0.28.11, default and `Options.FMA`; one core |

Notes on the files. `counters-65536-new-and-fftw.txt` starts with the
transform 1,024 counters of the new binary discarded by mistake (the table
that follows is 65,536). `fft2-and-fma-first-cut.txt` was taken before 2,048
points were kept on the AVX-512 route; only its FFT2 half (64² to 1024², all
below the changed lengths' limit and not touching 2,048) is used.

## Result

go-fft ÷ FFTW, median of 7 interleaved rounds:

| complex | v0.28.11 vs FFTW AVX-512 build | this change vs FFTW AVX-512 build | v0.28.11 vs FFTW AVX2 build | this change vs FFTW AVX2 build |
|--:|--:|--:|--:|--:|
| 256 | 1.62 | 1.29 | 1.43 | 1.13 |
| 1,024 | 1.48 | 1.31 | 1.31 | 1.14 |
| 4,096 | 1.13 | 1.06 | 1.00 | 0.93 |
| 65,536 | 0.985 | 0.984 | 0.88 | 0.87 |
| 1,048,576 | 1.09 | 1.07 | 1.07 | 1.07 |

(The first column divides by the FFTW AVX-512 build and the third by the AVX2
build, so v0.28.11's 1,024 is 1.48 and 1.31: FFTW's AVX-512 plan is 12% faster
than its AVX2 one there.)

Counters per complex 1,024 transform (user mode):

| | go-fft v0.28.11 | go-fft, AVX2 route | FFTW AVX-512 build | FFTW AVX2 build |
|:--|--:|--:|--:|--:|
| cycles | 5,961 | 5,166 | 4,071 | 4,471 |
| instructions | 13,922 | 18,437 | 12,102 | 17,588 |
| IPC | 2.34 | 3.57 | 2.97 | 3.93 |
| backend-stalled dispatch slots | 21,040 | 12,638 | 10,744 | 8,523 |
| fp scheduler token stalls | 2,947 | 809 | 953 | 600 |
| L1D fills from L2 | 1,806 | 1,607 | 912 | 717 |
| 64-byte-split loads (`ma64`) | 4,039 | 406 | 1,320 | 0 |
| store-to-load forwards (`ls_stlf`) | 22 | 613 | 398 | 521 |
| 4 KB / other STLI failures | 133 | 273 | 117 | 323 |
| DTLB misses | 0 | 1 | 0 | 0 |

Complex 65,536 (this change, user mode): go-fft 755,138 cycles, 1,929,667
instructions, IPC 2.56; FFTW AVX-512 build 756,357 cycles, 1,947,258
instructions, IPC 2.58; FFTW AVX2 build 863,714 cycles, 2,948,050
instructions, IPC 3.41. Parity with the AVX-512 build, 13% faster than the AVX2
one.
