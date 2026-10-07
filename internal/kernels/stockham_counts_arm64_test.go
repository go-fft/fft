package kernels

import (
	"strings"
	"testing"
)

// TestStockhamPassRefusesNonPositiveCounts: ido = -2, l1 = -3 multiply to a
// positive length that passes the bound checks, and the split final pass
// takes l1 without a twiddle table to fail on. Both entry points refuse such
// counts before any kernel runs, as the amd64 StockhamPass has since v0.18.0.
func TestStockhamPassRefusesNonPositiveCounts(t *testing.T) {
	buf := make([]complex128, 256)
	expect := func(name, prefix string, f func()) {
		t.Helper()
		defer func() {
			msg, _ := recover().(string)
			if !strings.HasPrefix(msg, prefix) {
				t.Errorf("%s: recovered %q, want a %q panic", name, msg, prefix)
			}
		}()
		f()
	}
	for _, c := range []struct{ ido, l1 int }{{-2, -3}, {0, 4}, {4, 0}, {1, -8}} {
		expect("StockhamPass", "kernels: StockhamPass:", func() {
			StockhamPass(4, c.ido, c.l1, buf, buf, buf, false, false)
		})
		for _, mode := range []uint8{splitIn, splitOut, splitBoth} {
			expect("StockhamPassLayout", "kernels: StockhamPassLayout:", func() {
				StockhamPassLayout(mode, 4, c.ido, c.l1, buf, buf, buf, false, false)
			})
		}
	}
}
