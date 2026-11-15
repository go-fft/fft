//go:build !arm64

package fft

// dct4Fused64Route is false off arm64: the fused kernel is measured there only.
const dct4Fused64Route = false

// The pair kernels are NEON: off arm64 the plan is never fused and these are
// not reached.
func dct4Pair64(dst0, dst1, src0, src1 []float64, f float64)      { panic("unreachable") }
func dct4Pair64Mixed(dst0, dst1, src0, src1 []float32, f float64) { panic("unreachable") }
