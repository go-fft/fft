package kernels

import (
	"slices"
	"testing"
)

// TestStockhamSplitModesComposite: which transforms keep their data split
// (Round 24 opened it to radix 3 and 5 before the last pass and to a radix-8
// last pass over any number of blocks).
func TestStockhamSplitModesComposite(t *testing.T) {
	for _, c := range []struct {
		r, ido, l1 []int
		want       []uint8
	}{
		// 1000 odd first: 5·5·5·8, l1 = 125 at the end.
		{[]int{5, 5, 5, 8}, []int{200, 40, 8, 1}, []int{1, 5, 25, 125}, []uint8{1, 2, 2, 3}},
		// 120: 3·5·8.
		{[]int{3, 5, 8}, []int{40, 8, 1}, []int{1, 3, 15}, []uint8{1, 2, 3}},
		// pocketfft order: the last pass is radix 5, which has no split kernel.
		{[]int{8, 5, 5, 5}, []int{125, 25, 5, 1}, []int{1, 8, 40, 200}, []uint8{0, 0, 0, 0}},
		// An odd ido before the last pass.
		{[]int{2, 5, 5}, []int{25, 5, 1}, []int{1, 2, 10}, []uint8{0, 0, 0}},
		// A radix without any NEON kernel (11) before the last pass.
		{[]int{11, 4}, []int{4, 1}, []int{1, 11}, []uint8{0, 0}},
		// One pass only.
		{[]int{8}, []int{1}, []int{1}, []uint8{0}},
	} {
		if got := StockhamSplitModes(c.r, c.ido, c.l1); !slices.Equal(got, c.want) {
			t.Errorf("StockhamSplitModes(%v, %v, %v) = %v, want %v", c.r, c.ido, c.l1, got, c.want)
		}
	}
}
