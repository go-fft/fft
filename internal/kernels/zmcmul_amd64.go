package kernels

// The AVX2 pointwise complex product (zmcmul_amd64.s, Round 36):
// genZmCMulFile in asmgen/amd64/gen.go.

//go:noescape
func zmCMulAVX2(dst, a, b *complex128, quads int)

// zmCMulBulk runs CMulTo's points four at a time on zmCMulAVX2 and returns
// how many it wrote (a multiple of four), 0 without AVX2 or below four
// points. len(a) and len(b) are len(dst).
func zmCMulBulk(dst, a, b []complex128) int {
	q := len(dst) / 4
	if !UseStockhamAVX2 || q == 0 {
		return 0
	}
	zmCMulAVX2(&dst[0], &a[0], &b[0], q)
	return 4 * q
}
