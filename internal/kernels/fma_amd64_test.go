package kernels

import (
	"reflect"
	"strings"
	"testing"
)

// TestFMAEntryPointsRefuse: the FMA entry points check what their default
// counterparts check, panic with the same messages, and report false when
// the AVX2 kernels are off. No kernel runs here, so it holds on any amd64 CPU.
func TestFMAEntryPointsRefuse(t *testing.T) {
	c := make([]complex128, 64)
	panics := func(name, want string, f func()) {
		t.Helper()
		defer func() {
			if r, _ := recover().(string); !strings.Contains(r, want) {
				t.Errorf("%s: panic %q, want %q", name, r, want)
			}
		}()
		f()
	}
	panics("StockhamPassFMA", "ido and l1 must be at least 1", func() { StockhamPassFMA(4, 4, 0, c, c, c, false, false) })
	panics("StockhamPassLayoutFMA", "no split kernel", func() { StockhamPassLayoutFMA(splitOut, 4, 3, 1, c, c, c, false, false) })
	panics("StockhamBatchPassFMA", "need w >= 1", func() { StockhamBatchPassFMA(4, 4, 1, c, c, c, 0, 1, 1, false) })
	panics("StockhamStridedFMA", "strides must be >= 0", func() { StockhamStridedFMA(4, c, c, c, 4, 1, -1, 16, 4, 16, 0, true, true, false, false) })
	panics("StockhamLastRunFMA", "os and gap must be >= 0", func() { StockhamLastRunFMA(4, c, c, -1, 1, 4, 4, false, false) })
	if StockhamBatchPassFMA(7, 2, 1, c, c, c, 1, 1, 1, false) || StockhamStridedFMA(8, c, c, c, 3, 1, 4, 16, 4, 16, 0, true, true, false, false) ||
		StockhamLastRunFMA(8, c, c, 4, 1, 3, 4, false, false) {
		t.Error("an FMA entry point ran a shape it has no kernel for")
	}
	defer func(a bool) { UseStockhamAVX2 = a }(UseStockhamAVX2)
	UseStockhamAVX2 = false
	if StockhamPassFMA(4, 4, 1, c, c, c, false, false) || StockhamPassLayoutFMA(splitNone, 4, 4, 1, c, c, c, false, false) ||
		StockhamStridedFMA(4, c, c, c, 4, 1, 4, 16, 4, 16, 0, true, true, false, false) || StockhamLastRunFMA(4, c, c, 4, 1, 4, 4, false, false) {
		t.Error("an FMA entry point ran with the AVX2 kernels off")
	}
}

// TestFMAKernelSelection: the selectors return the FMA twins at both widths,
// and the default final-pass kernels where a radix has no twin.
func TestFMAKernelSelection(t *testing.T) {
	same := func(a, b any) bool { return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer() }
	if !same(stockhamPassFMAKernel(8, 512), skPass8AVX512FMA) || !same(stockhamPassFMAKernel(8, 256), skPass8AVX2FMA) {
		t.Error("stockhamPassFMAKernel")
	}
	if !same(stockhamLastFMAKernel(8, 512), skLast8AVX512FMA) || !same(stockhamLastFMAKernel(16, 256), skLast16AVX2FMA) ||
		!same(stockhamLastFMAKernel(4, 512), skLast4AVX512) || !same(stockhamLastFMAKernel(5, 256), skLast5AVX2) {
		t.Error("stockhamLastFMAKernel")
	}
	if !same(stockhamLastRunFMAKernel(8, 512), skLastRun8AVX512FMA) || !same(stockhamLastRunFMAKernel(8, 256), skLastRun8AVX2FMA) ||
		!same(stockhamLastRunFMAKernel(4, 256), skLastRun4AVX2) {
		t.Error("stockhamLastRunFMAKernel")
	}
	if !same(stockhamStridedFMAKernel(8, 512), skStrided8AVX512FMA) || !same(stockhamStridedFMAKernel(4, 256), skStrided4AVX2FMA) {
		t.Error("stockhamStridedFMAKernel")
	}
	b := uint8(intelSplit512Base)
	if !same(intelSplitKernelFMA(b+splitBoth, 8, 8, 1, true), intelSplit512R8SSInvFMA) || !same(intelSplitKernelFMA(b+splitOut, 4, 8, 1, false), intelSplit512R4ISFwdFMA) ||
		!same(intelSplitKernelFMA(splitIn, 8, 4, 1, true), skSplit8SIInvFMA) || !same(intelSplitKernelFMA(splitNone, 3, 4, 1, false), skSplit3IIFwdFMA) {
		t.Error("intelSplitKernelFMA")
	}
	// Radix 32 (Round 42): the in-frame kernels from l1·ido = 64.
	if !same(intelSplitKernelFMA(splitOut, 32, 32, 1, false), skSplit32ISFwdFMA) || !same(intelSplitKernelFMA(splitOut, 32, 64, 1, false), skSplit32ISFwdFMAF) ||
		!same(intelSplitKernelFMA(b+splitIn, 32, 8, 4, true), intelSplit512R32SIInvFMA) || !same(intelSplitKernelFMA(b+splitNone, 32, 8, 8, true), intelSplit512R32IIInvFMAF) {
		t.Error("intelSplitKernelFMA, radix 32")
	}
	for _, c := range []struct {
		wide, inverse bool
		k             *float64
		fn            skLastFn
	}{
		{false, false, &skFwd[0][0], skLast32AVX2FMA}, {false, true, &skInv[0][0], skLast32AVX2FMA},
		{true, false, &sk512Fwd[0][0], skLast32AVX512FMA}, {true, true, &sk512Inv[0][0], skLast32AVX512FMA},
	} {
		if k, fn := r32LastKernel(c.wide, c.inverse); k != c.k || !same(fn, c.fn) {
			t.Errorf("r32LastKernel(%v, %v)", c.wide, c.inverse)
		}
	}
}
