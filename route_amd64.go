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
