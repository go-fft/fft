package kernels

import "testing"

// TestZmCMulBulkGate: the AVX2 product runs only where UseStockhamAVX2 says
// so and from four points on, and covers the largest multiple of four.
func TestZmCMulBulkGate(t *testing.T) {
	a := make([]complex128, 11)
	defer func(v bool) { UseStockhamAVX2 = v }(UseStockhamAVX2)
	want := 0
	if UseStockhamAVX2 {
		want = 8
	}
	for _, c := range []struct {
		on   bool
		n    int
		want int
	}{{true, 11, want}, {true, 3, 0}, {true, 0, 0}, {false, 11, 0}} {
		UseStockhamAVX2 = c.on && want > 0
		if got := zmCMulBulk(a[:c.n], a[:c.n], a[:c.n]); got != c.want {
			t.Errorf("AVX2 %v, %d points: %d run on the kernel, want %d", c.on, c.n, got, c.want)
		}
	}
}
