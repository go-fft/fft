//go:build !amd64

package fft

// dct4F32Route is false here: the float32 pair and quad kernels are AVX2+FMA3
// (amd64) for now.
func dct4F32Route(n, typ int) bool { return false }

func dct4n64F32Pair(dst0, dst1, src0, src1 []float32, f float32) {
	panic("fft: no float32 pair kernel on this platform")
}

func dct4n64F32Quad(dst, src *[4][]float32, f float32) {
	panic("fft: no float32 quad kernel on this platform")
}
