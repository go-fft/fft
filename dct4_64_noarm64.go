//go:build !arm64

package fft

// dct4FusedRoute is false off arm64: the fused kernels are measured there only.
const dct4FusedRoute = false

// The pair kernels are NEON. Off arm64 no plan is routed to them, but a test can
// force one (it sets fused), and then the pair is the scalar kernel twice, which
// keeps every line of the fused path covered on every platform.
func dct4Pair(n int, dst0, dst1, src0, src1 []float64, f float64) {
	dct4Single(n, dst0, src0, f)
	dct4Single(n, dst1, src1, f)
}

func dct4PairMixed(n int, dst0, dst1, src0, src1 []float32, f float64) {
	dct4SingleMixed(n, dst0, src0, f)
	dct4SingleMixed(n, dst1, src1, f)
}
