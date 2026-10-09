package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestZnFrameFallsBack: plans built to take the frame kernels still
// transform, on the pool's buffer, when the kernels cannot run (the AVX2
// kernels turned off after the plans were built).
func TestZnFrameFallsBack(t *testing.T) {
	if !kernels.UseStockhamAVX2 {
		t.Skip("no AVX2")
	}
	defer znSetMax(256, 512)()
	defer func(a bool) { kernels.UseStockhamAVX2 = a }(kernels.UseStockhamAVX2)
	for _, n := range []int{64, 256} {
		kernels.UseStockhamAVX2 = true
		p := NewPlan(n)
		if p.sk.zn == nil {
			t.Fatalf("%d points did not take a frame kernel", n)
		}
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64(i%7), float64(i%5)-2)
		}
		want := make([]complex128, n)
		p.FFT(want, src)
		kernels.UseStockhamAVX2 = false
		got := make([]complex128, n)
		p.FFT(got, src)
		for i := range got {
			if d := got[i] - want[i]; math.Abs(real(d)) > 1e-12 || math.Abs(imag(d)) > 1e-12 {
				t.Fatalf("n=%d [%d] = %v, want %v", n, i, got[i], want[i])
			}
		}
	}
}
