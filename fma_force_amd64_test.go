package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// forceFMA builds FMA plans for the rest of the test whatever the CPU: on
// one without FMA3 the kernels, FMA and default alike, are turned off, so
// that every pass runs its Go reference and none faults. The default
// arithmetic stays unfused meanwhile (defaultFuses off), so the tests that
// compare an FMA plan with "the default" keep comparing it with the unfused
// plan; a test of the fused default turns defaultFuses back on after calling
// forceFMA.
func forceFMA(t *testing.T) {
	t.Helper()
	a, b, c, d := fmaAvailable, kernels.UseStockhamAVX2, kernels.UseStockhamBatchAVX2, defaultFuses
	t.Cleanup(func() {
		fmaAvailable, kernels.UseStockhamAVX2, kernels.UseStockhamBatchAVX2, defaultFuses = a, b, c, d
		forgetFMAPlans()
	})
	fmaAvailable, defaultFuses = true, false
	if !kernels.UseFMA {
		kernels.UseStockhamAVX2, kernels.UseStockhamBatchAVX2 = false, false
	}
	forgetFMAPlans()
}
