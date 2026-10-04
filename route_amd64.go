package fft

import (
	"math"

	"github.com/go-fft/fft/internal/kernels"
)

// pow2StockhamMaxDefault routes every power of two to the Stockham engine on
// amd64 when its AVX2 (or AVX-512) pass kernels run.
//
// Until 2026-10-04 the AVX2 route stopped at 4096: measured then (pow2 kernel
// time ÷ Stockham time, 2026-09-30), Haswell and Cascade Lake lost above it
// and Zen 3 lost AT 4096. Those Stockham times were inflated by L1 set
// conflicts between dst and the scratch buffer (see offTheSets); with the
// scratch placed off dst's sets, the same comparison reads (2026-10-04):
//
//	n        4096  8192  16384  32768  65536  2^17  2^18  2^19  2^20
//	Zen 3    1.50  1.58  1.94   1.50   1.81   1.38  1.81  1.96  2.51
//	Haswell  1.76  1.05  1.65   1.04   1.16   0.82  1.15  1.23  1.24
//
// Stockham wins at every size on Zen 3, and on Haswell everywhere but 2^17;
// the geometric mean of the two is above 1 at every size. Without AVX2 the
// Stockham passes run scalar, and lost to the pow2 kernel's SSE2 butterflies
// at every size when measured (2026-09-30, before the scratch fix; not
// re-measured, since every host at hand has AVX2).
func pow2StockhamMaxDefault() int {
	return pow2StockhamMaxAMD64(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
}

// With AVX-512 the route stopped at 16384 (Cascade Lake, pow2 kernel time ÷
// Stockham time, 2026-10-01: 2.33 at 4096, 0.86–1.23 above 16384). Above 4096
// those powers of two were factored with rule B. Radix 8 as far as it goes
// (rule A, see r8MaxPow2AMD64) is 10–38% faster at every size there, and with
// it and the scratch off dst's sets the Stockham engine
// beats the pow2 kernel at every size on the same idle host (2026-10-04,
// ns per point):
//
//	n            32768  65536  2^17   2^18   2^19   2^20
//	Stockham A    6.18   9.00  10.50  12.72  25.49  28.67
//	pow2 kernel   9.91  13.18  15.30  24.41  33.31  36.66
func pow2StockhamMaxAMD64(avx2, avx512 bool) int {
	switch {
	case avx2 || avx512:
		return math.MaxInt
	}
	return 0
}

// r8MaxPow2Default keeps radix-8 passes for powers of two up to 4096 with the
// AVX2 passes, and for every power of two with the AVX-512 passes.
func r8MaxPow2Default() int {
	return r8MaxPow2AMD64(kernels.UseStockhamAVX512)
}

// r8MaxPow2AMD64 is r8MaxPow2Default's choice. Three rules were timed at every
// 2^e from 256 to 2^20, with the scratch off dst's sets (2026-10-04): A, radix
// 8 as far as it goes; B, radix 4 with a radix-2 pass for an odd exponent; C,
// radix 4 with one radix-8 pass for an odd exponent. Time over the best of
// the three, geometric mean (worst):
//
//	                    A to 4096, B   A to 1024, B   A everywhere
//	Haswell (AVX2)      1.090 (1.22)   1.124 (1.24)   1.000 (1.00)
//	Zen 3 (AVX2)        1.038 (1.31)   1.008 (1.04)   1.295 (1.71)
//	Cascade (AVX-512)   1.239 (1.62)   1.290 (1.62)   1.000 (1.00)
//
// The two AVX2 CPUs disagree from 2048 up: radix 8 wins every size on Haswell
// and lost every size on Zen 3. Most of Zen 3's loss was the radix-8 pass
// reading its seven twiddles as seven separate streams: with the twiddles
// laid out as one stream (see kernels.StockhamTwiddles), radix 8 ÷ radix 4 on
// Zen 3 went from 1.18 to 1.02 at 2048, 1.37 to 1.19 at 4096, 1.56–1.64 to
// 1.20–1.30 at 8192–32768 and to 0.98–1.06 above (2026-10-04). Over both, A to 4096 then B scores 1.064 (worst 1.31) — the best with
// A to 1024 (1.065, worst 1.24) — and is kept; A everywhere would be 1.138
// (worst 1.71). With AVX-512, radix 8 wins at every size.
func r8MaxPow2AMD64(avx512 bool) int {
	if avx512 {
		return math.MaxInt
	}
	return 4096
}

// pow2OneRadix8Max is 0 on amd64: above r8MaxPow2 a power of two takes rule B
// (see r8MaxPow2AMD64). Below it, timed at 2^5..2^13 on Haswell, Zen 3 and
// Cascade Lake (2026-09-30), rule A scored 1.082 over the best, C 1.129, B
// 1.190.
const pow2OneRadix8Max = 0
