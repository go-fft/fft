package kernels

import (
	"reflect"
	"testing"
)

// TestStockhamWidth pins the kernel choice for every combination, including
// the AVX-512 branches a CI runner without AVX-512 never executes.
func TestStockhamWidth(t *testing.T) {
	for _, c := range []struct {
		r, ido        int
		avx2, wide512 bool
		want          int
	}{
		{8, 1, true, true, 512}, {4, 4, true, true, 512}, {2, 64, true, true, 512},
		{8, 2, true, true, 256}, {4, 3, true, true, 256}, // fewer than four points: AVX2
		{3, 64, true, true, 256}, {5, 1, true, true, 256}, // no 512-bit kernel for 3, 5
		{8, 2, false, true, 0}, {8, 8, false, true, 512},
		{4, 64, true, false, 256}, {8, 1, true, false, 256},
		{7, 64, true, true, 0}, {11, 64, true, false, 0}, // no kernel at all
		{4, 64, false, false, 0},
	} {
		if got := stockhamWidth(c.r, c.ido, c.avx2, c.wide512); got != c.want {
			t.Errorf("stockhamWidth(r=%d, ido=%d, avx2=%v, wide512=%v) = %d, want %d", c.r, c.ido, c.avx2, c.wide512, got, c.want)
		}
	}
}

// TestSKConst512Rows: every 512-bit row is the 256-bit row twice, and the
// added row flips the real lanes only.
func TestSKConst512Rows(t *testing.T) {
	for _, inverse := range []bool{false, true} {
		w, n := newSKConst512(inverse), newSKConst(inverse)
		for r := range n {
			for i := range w[r] {
				if w[r][i] != n[r][i%4] || (w[r][i] == 0 && (1/w[r][i] > 0) != (1/n[r][i%4] > 0)) {
					t.Fatalf("inverse=%v row %d lane %d: %v vs %v", inverse, r, i, w[r][i], n[r][i%4])
				}
			}
		}
		for i, v := range w[8] {
			if v != 0 || (1/v < 0) != (i%2 == 0) {
				t.Errorf("add-sub row lane %d = %v, want -0 on real lanes and +0 on imaginary", i, v)
			}
		}
	}
}

// TestStockhamKernels checks the selection for both widths and directions:
// the table's direction row and the kernels for every radix that has them.
func TestStockhamKernels(t *testing.T) {
	same := func(a, b any) bool { return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer() }
	for _, c := range []struct {
		w       int
		inverse bool
		k       *float64
		pass    [9]skPassFn
		last    [9]skLastFn
	}{
		{256, false, &skFwd[0][0], skPass256, skLast256},
		{256, true, &skInv[0][0], skPass256, skLast256},
		{512, false, &sk512Fwd[0][0], skPass512, skLast512},
		{512, true, &sk512Inv[0][0], skPass512, skLast512},
	} {
		for _, r := range []int{2, 4, 8} {
			k, pass, last := stockhamKernels(r, c.w, c.inverse)
			if k != c.k || !same(pass, c.pass[r]) || !same(last, c.last[r]) {
				t.Errorf("stockhamKernels(%d, %d, %v) picked the wrong table or kernels", r, c.w, c.inverse)
			}
		}
	}
}
