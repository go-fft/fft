package fft

import "github.com/go-fft/fft/internal/kernels"

// pow2StockhamMaxDefault routes powers of two up to 4096 to the Stockham
// engine on amd64 when its AVX2 pass kernels run, and larger ones to the
// iterative pow2 kernel. Both use AVX2 there. Measured on real hardware
// (2026-09-30, pow2 kernel time ÷ Stockham time):
//
//	n            64    256   1024  4096  16384  65536  262144  2^20
//	Haswell      1.28  1.63  1.35  1.28  0.92   0.97   0.81    1.07
//	Zen 3        1.51  1.51  1.35  0.95  1.58   1.38   1.40    1.43
//	Cascade Lake 1.57  1.86  1.63  1.22  1.04   1.00   1.08    0.98
//
// Stockham wins up to 4096 everywhere but Zen 3's 4096; above it the three
// disagree, and the pow2 kernel is kept. Without AVX2 the Stockham passes run
// scalar and lose to the pow2 kernel's SSE2 butterflies at every size.
func pow2StockhamMaxDefault() int { return pow2StockhamMaxAMD64(kernels.UseStockhamAVX2) }

func pow2StockhamMaxAMD64(avx2 bool) int {
	if avx2 {
		return 4096
	}
	return 0
}

// r8MaxPow2Arch keeps radix-8 passes for powers of two up to 4096 on amd64.
// With the AVX2 passes, radix 8 beat radix 4 at every power of two from 256 to
// 4096 on Haswell (0.80–0.87× the time at 2048/4096) and Cascade Lake
// (0.74–0.91×), but not on Zen 3 at 2048 and 4096 (1.36×, 1.42×), where the
// radix-8 pass's 23 power-of-two-strided streams contend for L1 sets. Two of
// the three CPUs prefer radix 8 and the geometric mean is neutral at 2048/4096,
// so it stays; Zen 3's loss is recorded in BENCHMARKS.md. (2026-09-30)
const r8MaxPow2Arch = 4096

// pow2OneRadix8Max is unused on amd64 below r8MaxPow2 (rule A factors those);
// above it powers of two take the pow2 kernel. Timed there, rule A — radix 8
// as far as it goes — scored 1.082 over the best (geometric mean, Haswell, Zen 3,
// Cascade Lake, 2^5..2^13), radix 4 plus one radix 8 for an odd exponent 1.129,
// radix 4 plus a radix-2 pass 1.190.
const pow2OneRadix8Max = 0
