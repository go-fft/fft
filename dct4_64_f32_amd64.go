package fft

// dct4F32Asm says whether a DCTPlan32 of type 4 and length 64 runs the float32
// AVX2+FMA3 kernels (dct4_64_f32_amd64.s): fmaAvailable. A plan reads it when it is
// built.
var dct4F32Asm = fmaAvailable

// dct4F32Route reports whether a new DCTPlan32 of the type and length routes its
// pair and quad forms to the kernels.
func dct4F32Route(n, typ int) bool { return n == 64 && typ == 4 && dct4F32Asm }

// The kernels read all inputs before the first store: dst[i] may alias src[i].

//go:noescape
func dct4n64Y32Pair(dst0, dst1, src0, src1 *float32, w *dct4n64Y32Tab)

//go:noescape
func dct4n64Y32PairS(dst0, dst1, src0, src1 *float32, w *dct4n64Y32Tab, f float32)

//go:noescape
func dct4n64Y32Quad(dst0, dst1, dst2, dst3, src0, src1, src2, src3 *float32, w *dct4n64Y32Tab)

//go:noescape
func dct4n64Y32QuadS(dst0, dst1, dst2, dst3, src0, src1, src2, src3 *float32, w *dct4n64Y32Tab, f float32)

func dct4n64F32Pair(dst0, dst1, src0, src1 []float32, f float32) {
	if f != 1 {
		dct4n64Y32PairS(&dst0[0], &dst1[0], &src0[0], &src1[0], &dct4n64Y32, f)
		return
	}
	dct4n64Y32Pair(&dst0[0], &dst1[0], &src0[0], &src1[0], &dct4n64Y32)
}

func dct4n64F32Quad(dst, src *[4][]float32, f float32) {
	if f != 1 {
		dct4n64Y32QuadS(&dst[0][0], &dst[1][0], &dst[2][0], &dst[3][0], &src[0][0], &src[1][0], &src[2][0], &src[3][0], &dct4n64Y32, f)
		return
	}
	dct4n64Y32Quad(&dst[0][0], &dst[1][0], &dst[2][0], &dst[3][0], &src[0][0], &src[1][0], &src[2][0], &src[3][0], &dct4n64Y32)
}
