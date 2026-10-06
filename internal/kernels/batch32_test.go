//go:build amd64 || arm64

package kernels

import (
	"strings"
	"testing"
)

// TestStockhamBatchPass32RejectsInconsistentLayouts: the float32 batch
// kernels trust w, sIn and sOut, so the wrapper refuses a layout its bound
// checks cannot vouch for, whether or not the kernels run, and bound-checks
// the last value each pass reads and writes before a kernel runs.
func TestStockhamBatchPass32RejectsInconsistentLayouts(t *testing.T) {
	buf := make([]complex64, 64)
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
				if !strings.HasPrefix(msg, "kernels: StockhamBatchPass32:") {
					t.Errorf("%+v: recovered %q, want the wrapper's panic", c, msg)
				}
			}()
			StockhamBatchPass32(4, c.ido, c.l1, buf, buf, buf, c.w, c.sIn, c.sOut, false)
		}()
	}
	if !UseStockhamBatch32 {
		t.Skip("no float32 batched kernels on this CPU")
	}
	// Radix 4, ido 2, l1 2: 16 points of 3 values, stride 5: the last value
	// is at 15·5+2 = 77; three twiddles.
	for _, c := range []struct {
		name       string
		cc, ch, tw int
	}{{"short cc", 77, 78, 3}, {"short ch", 78, 77, 3}, {"short tw", 78, 78, 2}} {
		func() {
			defer func() {
				err, _ := recover().(error)
				if err == nil || !strings.Contains(err.Error(), "index out of range") {
					t.Errorf("%s: recovered %v, want a bounds panic", c.name, err)
				}
			}()
			StockhamBatchPass32(4, 2, 2, make([]complex64, c.cc), make([]complex64, c.ch), make([]complex64, c.tw), 3, 5, 5, false)
		}()
	}
	if StockhamBatchPass32(7, 2, 2, buf, buf, buf, 1, 1, 1, false) {
		t.Error("radix 7 has no batched kernel")
	}
	if !StockhamBatchKernels32() {
		t.Error("StockhamBatchKernels32 false where UseStockhamBatch32 is on")
	}
}
