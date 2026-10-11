package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// znAVX2Route makes the plans built from now on take the route an AVX2 CPU
// that is not Intel's takes (Zen 3, where Round 32 measured the frame
// kernels), whatever this CPU is, and returns a func restoring the route. The
// frame kernels are AVX2 code and run the 256-bit layouts only, so on an
// Intel CPU with AVX-512 (the 512-bit split layout from 256 points, Intel's
// composite rules) kernels.ZnPlanFor rightly refuses most plans; the tests
// that hold the frame to the pool must build the plans it runs. The tables
// computed at init (splitTable, radix16Table, hswSplitFloor) are replaced
// too, not only the kernels' switches.
func znAVX2Route() func() {
	k512, s512, s256, u512 := kernels.UseStockhamAVX512, kernels.UseStockhamSplit512, kernels.UseStockhamSplit, kernels.UseUntangleAVX512
	st, r16, floor := splitTable, radix16Table, hswSplitFloor
	c16, c2, hsw := compRadix16On, comp2On, hswCompOn
	avx2 := kernels.UseStockhamAVX2
	kernels.UseStockhamAVX512, kernels.UseStockhamSplit512, kernels.UseStockhamSplit, kernels.UseUntangleAVX512 = false, false, avx2, false
	splitTable, radix16Table, hswSplitFloor = splitTableAMD64(avx2), radix16TableAMD64(avx2, false, false), 0
	compRadix16On, comp2On, hswCompOn = false, avx2, false
	return func() {
		kernels.UseStockhamAVX512, kernels.UseStockhamSplit512, kernels.UseStockhamSplit, kernels.UseUntangleAVX512 = k512, s512, s256, u512
		splitTable, radix16Table, hswSplitFloor = st, r16, floor
		compRadix16On, comp2On, hswCompOn = c16, c2, hsw
	}
}

// TestZnFrameFallsBack: plans built to take the frame kernels still
// transform, on the pool's buffer, when the kernels cannot run (the AVX2
// kernels turned off after the plans were built).
func TestZnFrameFallsBack(t *testing.T) {
	if !kernels.UseStockhamAVX2 {
		t.Skip("no AVX2")
	}
	defer znAVX2Route()()
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
