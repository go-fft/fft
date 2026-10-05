package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestCascadeKernelModes runs the identity check on the Go passes too.
func TestCascadeKernelModes(t *testing.T) {
	if testing.Short() {
		t.Skip("large transforms")
	}
	save := kernels.UseStockhamNEON
	defer func() { kernels.UseStockhamNEON = save }()
	kernels.UseStockhamNEON = false
	checkCascade(t, "go")
}
