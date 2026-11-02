package fft

// On Apple Silicon the 576-byte gap between dst and the scratch buffer
// (stockham.go, offTheSets) costs the large powers of two a fifth of their
// time. Whole transforms, Apple M4 Max, one core, 1-min load 7.3–7.9, eleven
// rotated rounds (benchmarks/results/arm64-m4-gap-20261102/g1.txt, gap in
// bytes, time ÷ the 576-byte time): 65536 points 0 0.80, 1088 0.93, 2112 0.84,
// 4032 0.81; 524288 points 0 0.83, 2112 0.83, 4032 0.83; 2^20 points 0 0.88,
// 4032 0.87. From 4096 to 32768 points every gap is within 2% of 576 except
// none (0), which loses 3–7% at 4096. 4032 bytes (64 short of a 4 KB period)
// keeps the buffers off one set and wins everywhere measured. Neoverse-N1 and
// the other architectures, where 576 was tuned, keep it.
func init() { setGap = 4032 }
