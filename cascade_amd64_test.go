package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestCascadeKernelModes runs the identity check with each kernel family the
// machine has forced: AVX-512, AVX2 alone, and the Go passes.
func TestCascadeKernelModes(t *testing.T) {
	if testing.Short() {
		t.Skip("large transforms")
	}
	a2, a512 := kernels.UseStockhamAVX2, kernels.UseStockhamAVX512
	defer func() { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a2, a512 }()
	if a512 {
		checkCascade(t, "avx512")
	}
	kernels.UseStockhamAVX512 = false
	if a2 {
		checkCascade(t, "avx2")
	}
	kernels.UseStockhamAVX2 = false
	checkCascade(t, "go")
}

// TestCascadeMinAMD64 pins the threshold for every combination.
func TestCascadeMinAMD64(t *testing.T) {
	for _, c := range []struct {
		avx512, intel bool
		want          int
	}{
		{true, true, 1 << 16}, {true, false, 0}, {false, true, 0}, {false, false, 0},
	} {
		if got := cascadeMinAMD64(c.avx512, c.intel); got != c.want {
			t.Errorf("cascadeMinAMD64(%v, %v) = %d, want %d", c.avx512, c.intel, got, c.want)
		}
	}
}
