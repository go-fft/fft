package kernels

// The AVX2 pointwise complex product (zmcmul_amd64.s, Round 36):
// genZmCMulFile in asmgen/amd64/gen.go.

//go:noescape
func zmCMulAVX2(dst, a, b *complex128, quads int)

// ZmCMulOn lets CMulTo run zmCMulAVX2: on where the CPU is not Intel's, the
// only vendor Round 36 measured it on (Zen 3). Its results are the Go
// loop's bits either way. A variable so the tests can take both routes on
// any machine.
var ZmCMulOn = !IntelCPU

// zmCMulBulk runs CMulTo's points four at a time on zmCMulAVX2 and returns
// how many it wrote (a multiple of four): 0 without AVX2, with ZmCMulOn
// off, or below four points. len(a) and len(b) are len(dst).
func zmCMulBulk(dst, a, b []complex128) int {
	q := len(dst) / 4
	if !UseStockhamAVX2 || !ZmCMulOn || q == 0 {
		return 0
	}
	zmCMulAVX2(&dst[0], &a[0], &b[0], q)
	return 4 * q
}
