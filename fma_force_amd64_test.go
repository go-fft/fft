package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// forceFMA builds FMA plans for the rest of the test whatever the CPU: on
// one without FMA3 the kernels, FMA and default alike, are turned off, so
// that every pass runs its Go reference and none faults.
func forceFMA(t *testing.T) {
	t.Helper()
	a, b, c := fmaAvailable, kernels.UseStockhamAVX2, kernels.UseStockhamBatchAVX2
	t.Cleanup(func() {
		fmaAvailable, kernels.UseStockhamAVX2, kernels.UseStockhamBatchAVX2 = a, b, c
		forgetFMAPlans()
	})
	fmaAvailable = true
	if !kernels.UseFMA {
		kernels.UseStockhamAVX2, kernels.UseStockhamBatchAVX2 = false, false
	}
	forgetFMAPlans()
}
