package kernels

import (
	"strings"
	"testing"
)

// TestStockhamBatchPassRejectsInconsistentLayouts: the batch kernels trust w,
// sIn and sOut, so the wrapper refuses a layout its bound check cannot vouch
// for, whether or not the CPU has AVX2.
func TestStockhamBatchPassRejectsInconsistentLayouts(t *testing.T) {
	buf := make([]complex128, 64)
	for _, c := range []struct{ ido, l1, w, sIn, sOut int }{
		{1, 2, 0, 4, 4},  // empty batch
		{1, 2, -1, 4, 4}, // negative width
		{1, 2, 4, 3, 4},  // input stride below the width
		{1, 2, 4, 4, 2},  // output stride below the width
		{0, 2, 2, 2, 2},  // no points per block
		{1, 0, 2, 2, 2},  // no blocks
	} {
		func() {
			defer func() {
				msg, _ := recover().(string)
				if !strings.HasPrefix(msg, "kernels: StockhamBatchPass:") {
					t.Errorf("%+v: recovered %q, want the wrapper's panic", c, msg)
				}
			}()
			StockhamBatchPass(4, c.ido, c.l1, buf, buf, buf, c.w, c.sIn, c.sOut, false)
		}()
	}
}
