//go:build !amd64

package kernels

import "testing"

// TestFMANoKernels: without amd64's FMA kernels every FMA entry point
// reports false, so the fft package runs its Go reference passes.
func TestFMANoKernels(t *testing.T) {
	c := make([]complex128, 16)
	if UseFMA || StockhamPassFMA(4, 4, 1, c, c, c, false, false) || StockhamPassLayoutFMA(1, 4, 4, 1, c, c, c, false, false) ||
		StockhamBatchPassFMA(4, 4, 1, c, c, c, 1, 1, 1, false) || StockhamStridedFMA(4, c, c, c, 4, 1, 4, 16, 4, 16, 0, true, true, false, false) ||
		StockhamLastRunFMA(4, c, c, 4, 1, 4, 4, false, false) || ZnPlanForWideFMA([]int{4, 4}, []uint8{0, 0}, c, c, nil, nil, false) != nil {
		t.Fatal("an FMA entry point ran without FMA kernels")
	}
}
