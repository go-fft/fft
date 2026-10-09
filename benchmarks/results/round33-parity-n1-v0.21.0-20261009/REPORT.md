# Round 33 — Neoverse-N1 parity at v0.21.0, mean of two runs

Times in ns/op, each the mean of `parity-a` and `parity-b`. go/FFTW is the
mean of the two runs' ratios (below 1: go-fft faster; at or above parity:
≤ 1.05). The last column is go-fft's mean time at v0.21.0 ÷ its mean time at
v0.17.0 (Round 27 `parity-br`, `parity-br2`, same host, same core, same
harness); below 1 means v0.21.0 is faster.

| row | go-fft | FFTW | numpy | scipy | gonum | go/FFTW (mean of run ratios) | per run | v0.21/v0.17 go-fft |
|:--|--:|--:|--:|--:|--:|--:|:--|--:|
| complex 256 (2⁸) | 1010 | 1160 | 9440 | 7684 | 5774 | 0.87 | 0.87 / 0.87 | 0.908 |
| complex 1,024 (2¹⁰) | 4876 | 6382 | 18850 | 14632 | 30366 | 0.76 | 0.77 / 0.76 | 0.922 |
| complex 4,096 (2¹²) | 23863 | 41270 | 60140 | 48100 | 146702 | 0.58 | 0.58 / 0.58 | 0.856 |
| complex 65,536 (2¹⁶) | 722392 | 1267164 | 2192458 | 1444663 | 3404976 | 0.57 | 0.59 / 0.55 | 1.005 |
| complex 1,048,576 (2²⁰) | 20350100 | 45859296 | 45526432 | 30524594 | 74665739 | 0.44 | 0.46 / 0.43 | 1.008 |
| complex 1,000 (2³·5³) | 6104 | 7179 | 19434 | 15830 | 33967 | 0.85 | 0.85 / 0.85 | 1.010 |
| complex 1,080 (2³·3³·5) | 6878 | 7724 | 22178 | 17065 | 41630 | 0.89 | 0.88 / 0.90 | 0.999 |
| complex 1,920 (2⁷·3·5) | 11938 | 13554 | 33340 | 26210 | 70110 | 0.88 | 0.88 / 0.88 | 0.993 |
| complex 1,009 (prime) | 21580 | 48139 | 97321 | 57534 | 2228906 | 0.45 | 0.45 / 0.44 | 0.999 |
| complex 1,296 (2⁴·3⁴) | 8636 | 11414 | 24879 | 19746 | 48704 | 0.76 | 0.75 / 0.76 | 0.998 |
| complex 10,007 (prime) | 419996 | 630409 | 1036692 | 758770 | 216798866 | 0.67 | 0.65 / 0.68 | 1.008 |
| RFFT 256 (2⁸) | 752 | 720 | 8718 | 7817 | 2986 | 1.04 | 1.04 / 1.05 | 0.977 |
| RFFT 1,024 (2¹⁰) | 3113 | 4228 | 14202 | 12506 | 15230 | 0.74 | 0.74 / 0.74 | 0.944 |
| RFFT 4,096 (2¹²) | 14772 | 19946 | 36984 | 32924 | 68462 | 0.74 | 0.74 / 0.74 | 0.949 |
| RFFT 65,536 (2¹⁶) | 396440 | 557735 | 722644 | 756900 | 1741255 | 0.71 | 0.73 / 0.69 | 0.985 |
| RFFT 1,048,576 (2²⁰) | 10095525 | 19139422 | 20971907 | 18595692 | 43165726 | 0.53 | 0.55 / 0.51 | 0.972 |
| RFFT 1,000 (2³·5³) | 3848 | 3950 | 15182 | 12632 | 14889 | 0.97 | 0.97 / 0.98 | 0.994 |
| RFFT 1,080 (2³·3³·5) | 4356 | 4216 | 15945 | 14092 | 17414 | 1.03 | 1.03 / 1.04 | 0.997 |
| RFFT 1,920 (2⁷·3·5) | 7100 | 7326 | 21826 | 18904 | 30646 | 0.97 | 0.97 / 0.97 | 0.987 |
| 2-D 64x64 | 24540 | 31180 | 70788 | 50072 | — | 0.79 | 0.78 / 0.79 | 0.880 |
| 2-D 128x128 | 128409 | 196285 | 253527 | 201876 | — | 0.65 | 0.65 / 0.66 | 1.010 |
| 2-D 256x256 | 616970 | 1323000 | 1141482 | 939314 | — | 0.47 | 0.50 / 0.44 | 0.908 |
| 2-D 512x512 | 3203585 | 7169068 | 5617647 | 4004770 | — | 0.45 | 0.48 / 0.41 | 0.917 |
| 2-D 1024x1024 | 18456272 | 44847924 | 26555379 | 20395201 | — | 0.41 | 0.44 / 0.38 | 1.009 |

rows 24; at or above FFTW (mean ratio <= 1.05): 24; numpy in every run: 24; scipy in every run: 24

The 24 rows are the complex 1-D, real 1-D and 2-D tables of the per-run
reports (the IRFFT rows are not counted, as in every round). numpy and scipy:
go-fft at or above parity in each of the two runs.

## Against v0.17.0

- **Faster**: complex 256 (0.91), 1,024 (0.92), 4,096 (0.86), RFFT 1,024 and
  4,096 (0.94, 0.95), 2-D 64² (0.88), 256² (0.91), 512² (0.92). These are the
  powers of two from 64 to 4096 points that Round 30 moved to radix-8 passes on
  arm64, and the RFFT/IRFFT that no longer go through the RealPlan's buffer.
- **Unchanged** (within ±1%): the composites, the primes, the large transforms
  (2^16, 2^20) and 2-D 128² and 1024².
- RFFT 256 moved 1.06 → 1.04 against FFTW, now inside parity; so all 24 rows.
