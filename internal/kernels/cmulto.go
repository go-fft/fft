package kernels

// CMulTo writes dst[i] = a[i]·b[i] for every i < len(dst), Go's complex
// product, and panics before writing anything if a or b is shorter than dst.
// dst may be a (in place); it must not otherwise overlap a or b. On amd64
// with AVX2, where ZmCMulOn (not Intel) says so, the points four at a time
// run on zmCMulAVX2 (zmcmul_amd64.s, Round 36), whose products, sums and
// differences are the GOAMD64=v1 scalar loop's, bit for bit; the rest, and
// every other CPU, run the scalar loop below. Bluestein's chirp and spectral
// products call it.
func CMulTo(dst, a, b []complex128) {
	n := len(dst)
	if len(a) < n || len(b) < n {
		// Slicing a[:n] would not catch it: a shorter slice with room in its
		// capacity reslices without a panic, and the kernel trusts n.
		panic("kernels: CMulTo operand shorter than dst")
	}
	a, b = a[:n], b[:n]
	for i := zmCMulBulk(dst, a, b); i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}
