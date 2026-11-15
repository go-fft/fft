//go:build !arm64 && !amd64

package fft

// dct4Fused64Route is false here: the fused scalar kernel is measured on arm64
// only (on amd64 the pooled path beats it, 60 ns against 75 ns on a Zen 4).
const dct4Fused64Route = false

// dct4PairRoute is false here: the pair kernels are NEON (arm64) and AVX2
// (amd64).
func dct4PairRoute() bool { return false }

// The pair kernels. No plan is routed to them here, but a test can force one (it
// sets fused and pair), and then the pair is the scalar kernel twice, which keeps
// every line of the fused path covered on every platform.
func dct4Pair64(dst0, dst1, src0, src1 []float64, f float64) {
	dct4Single64(dst0, src0, f)
	dct4Single64(dst1, src1, f)
}

func dct4Pair64Mixed(dst0, dst1, src0, src1 []float32, f float64) {
	dct4Single64Mixed(dst0, src0, f)
	dct4Single64Mixed(dst1, src1, f)
}
