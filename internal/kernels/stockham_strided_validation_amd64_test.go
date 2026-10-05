package kernels

import (
	"strings"
	"testing"
)

// TestStridedKernelsRejectNegativeStrides: StockhamLastRun and
// StockhamStrided bound-check the element their strides make the last one,
// which is the highest only when every stride is non-negative. A negative one
// is refused on every machine, before any kernel could run.
func TestStridedKernelsRejectNegativeStrides(t *testing.T) {
	buf := make([]complex128, 4096)
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
	for _, c := range []struct{ os, gap int }{{-1, 8}, {8, -1}} {
		expect("LastRun", "kernels: StockhamLastRun:", func() {
			StockhamLastRun(4, buf, buf, c.os, 1, 4, c.gap, false, false)
		})
	}
	for i := 0; i < 5; i++ {
		s := []int{16, 64, 16, 64, 1}
		s[i] = -1
		expect("Strided", "kernels: StockhamStrided:", func() {
			StockhamStrided(4, buf, buf, buf, 4, 1, s[0], s[1], s[2], s[3], s[4], true, true, false, false)
		})
	}
}
