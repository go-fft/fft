package kernels

import "testing"

// TestZmCMulBulkGate: the AVX2 product runs only where UseStockhamAVX2 and
// ZmCMulOn say so and from four points on, and covers the largest multiple
// of four.
func TestZmCMulBulkGate(t *testing.T) {
	a := make([]complex128, 11)
	defer func(v, o bool) { UseStockhamAVX2, ZmCMulOn = v, o }(UseStockhamAVX2, ZmCMulOn)
	avx2 := UseStockhamAVX2
	want := 0
	if avx2 {
		want = 8
	}
	for _, c := range []struct {
		avx2, on bool
		n        int
		want     int
	}{{true, true, 11, want}, {true, true, 3, 0}, {true, true, 0, 0}, {false, true, 11, 0}, {true, false, 11, 0}} {
		UseStockhamAVX2, ZmCMulOn = c.avx2 && avx2, c.on
		if got := zmCMulBulk(a[:c.n], a[:c.n], a[:c.n]); got != c.want {
			t.Errorf("AVX2 %v, ZmCMulOn %v, %d points: %d run on the kernel, want %d", c.avx2, c.on, c.n, got, c.want)
		}
	}
}

// TestZmCMulKernelMatchesScalar is TestCMulToMatchesScalar with ZmCMulOn
// on, so an Intel CPU, where it is off, still compares the kernel.
func TestZmCMulKernelMatchesScalar(t *testing.T) {
	defer func(o bool) { ZmCMulOn = o }(ZmCMulOn)
	ZmCMulOn = true
	zmCheckCMulTo(t)
}
