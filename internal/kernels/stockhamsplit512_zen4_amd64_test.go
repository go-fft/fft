package kernels

import "testing"

// TestStockhamSplitModes512: all zeros without AVX2 or AVX-512, the 512-bit
// modes with both; StockhamSplitModes512 is the rule on this machine.
func TestStockhamSplitModes512(t *testing.T) {
	r, ido := []int{4, 8}, []int{128, 16}
	for _, c := range [][2]bool{{false, true}, {true, false}, {false, false}} {
		for _, m := range splitModes512(c[0], c[1], r, ido) {
			if m != 0 {
				t.Errorf("avx2=%v avx512=%v: mode %d, want 0", c[0], c[1], m)
			}
		}
	}
	want := splitModes512(true, true, r, ido)
	if len(want) != 2 || want[0] == 0 || want[1] == 0 {
		t.Errorf("both: %v, want two 512-bit modes", want)
	}
	got := StockhamSplitModes512(r, ido, []int{1, 4})
	if exp := splitModes512(useAVX2, intelHasAVX512, r, ido); len(got) != 2 || got[0] != exp[0] || got[1] != exp[1] {
		t.Errorf("StockhamSplitModes512 = %v, want %v", got, exp)
	}
}
