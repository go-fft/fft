//go:build !amd64

package fft

// dct4F32Route is false here: the float32 pair and quad kernels are AVX2+FMA3
// (amd64) for now.
func dct4F32Route(n, typ int) bool { return false }

// With no assembly the kernels are a portable stand-in with the kernel's contract
// (every input read before the first store, one DCT-IV(64) per stream, scaled by
// f) in float64 arithmetic rounded to float32. No plan routes to them (quad is
// never set); the tests set it, so that the routing, the overlap rules and the
// arithmetic the kernel route must give are checked on every platform.

var dct4n64F32Plan = NewDCTPlan(64, 4)

func dct4n64F32Streams(dst, src [][]float32, f float32) {
	var in [4][64]float32
	for i := range src {
		copy(in[i][:], src[i][:64])
	}
	for i := range dst {
		x := make([]float64, 64)
		for j := range x {
			x[j] = float64(in[i][j])
		}
		y := dct4n64F32Plan.DCT(make([]float64, 64), x, NormBackward)
		for j := range y {
			dst[i][j] = float32(y[j] * float64(f))
		}
	}
}

func dct4n64F32Pair(dst0, dst1, src0, src1 []float32, f float32) {
	dct4n64F32Streams([][]float32{dst0, dst1}, [][]float32{src0, src1}, f)
}

func dct4n64F32Quad(dst, src *[4][]float32, f float32) {
	dct4n64F32Streams(dst[:], src[:], f)
}
