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
// too, not only the kernels' switches, and so is Round 35's radix-9 rule.
func znAVX2Route() func() {
	k512, s512, s256, u512 := kernels.UseStockhamAVX512, kernels.UseStockhamSplit512, kernels.UseStockhamSplit, kernels.UseUntangleAVX512
	st, r16, floor := splitTable, radix16Table, hswSplitFloor
	c16, c2, hsw, r9 := compRadix16On, comp2On, hswCompOn, csRadix9On
	avx2 := kernels.UseStockhamAVX2
	kernels.UseStockhamAVX512, kernels.UseStockhamSplit512, kernels.UseStockhamSplit, kernels.UseUntangleAVX512 = false, false, avx2, false
	splitTable, radix16Table, hswSplitFloor = splitTableAMD64(avx2), radix16TableAMD64(avx2, false, false), 0
	compRadix16On, comp2On, hswCompOn, csRadix9On = false, avx2, false, false
	return func() {
		kernels.UseStockhamAVX512, kernels.UseStockhamSplit512, kernels.UseStockhamSplit, kernels.UseUntangleAVX512 = k512, s512, s256, u512
		splitTable, radix16Table, hswSplitFloor = st, r16, floor
		compRadix16On, comp2On, hswCompOn, csRadix9On = c16, c2, hsw, r9
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

// TestCsZnFrameHostRoute is TestZnFrameMatchesPool on this machine's own
// route (Round 35): on an Intel CPU with AVX-512 the powers of two from 256
// points run 512-bit split passes and an AVX-512 final pass, and their frame
// plans (kernels.ZnPlanForWide) must give the pool's bits too, as must the
// radix-9 composites, for the transforms and the real transforms of twice
// their length, forward and inverse, in place and not.
func TestCsZnFrameHostRoute(t *testing.T) {
	if !kernels.UseStockhamAVX2 {
		t.Skip("no AVX2")
	}
	defer znSetMax(0, 0)()
	same := func(a, b []complex128) bool {
		for i := range a {
			if !sameBits(a[i], b[i]) {
				return false
			}
		}
		return true
	}
	wideTaken := 0
	for _, n := range []int{16, 64, 128, 256, 512, 18, 81, 162, 324, 192, 240, 288, 500} {
		znSetMax(0, 0)
		p0 := NewPlan(n)
		znForget(n)
		r0 := NewRealPlan(2 * n)
		znSetMax(kernels.ZnTwoPassMax, kernels.ZnThreePassMax)
		p := NewPlan(n)
		znForget(n)
		r := NewRealPlan(2 * n)
		znForget(n)
		if p.sk == nil || p0.sk.zn != nil || r0.half.sk.zn != nil {
			t.Fatalf("n=%d: no Stockham plan, or the pool plans took a frame", n)
		}
		if p.sk.zn != nil && p.sk.stages[0].wide && kernels.UseStockhamAVX512 {
			// The final pass at the width StockhamPass gives it.
			if !p.sk.zn.Wide() {
				t.Fatalf("n=%d: an AVX2 final pass where StockhamPass runs the AVX-512 one", n)
			}
			wideTaken++
		}
		src := cmplxSignal(n)
		for _, inverse := range []bool{false, true} {
			want, got := make([]complex128, n), make([]complex128, n)
			p0.sk.transform(want, src, inverse)
			p.sk.transform(got, src, inverse)
			in := append([]complex128(nil), src...)
			p.sk.transform(in, in, inverse)
			if !same(got, want) || !same(in, want) {
				t.Fatalf("n=%d inverse=%v: the frame differs from the pool (zn %v)", n, inverse, p.sk.zn != nil)
			}
		}
		x := make([]float64, 2*n)
		for i := range x {
			x[i] = float64(i%11) - 4.5
		}
		wantR := r0.RFFT(make([]complex128, n+1), x)
		gotR := r.RFFT(make([]complex128, n+1), x)
		wantI := r0.IRFFT(make([]float64, 2*n), wantR)
		gotI := r.IRFFT(make([]float64, 2*n), wantR)
		if !same(gotR, wantR) {
			t.Fatalf("RFFT %d: the frame differs from the pool", 2*n)
		}
		for i := range gotI {
			if math.Float64bits(gotI[i]) != math.Float64bits(wantI[i]) {
				t.Fatalf("IRFFT %d: [%d] differs", 2*n, i)
			}
		}
	}
	if kernels.UseStockhamAVX512 && kernels.UseStockhamSplit512 && wideTaken < 2 {
		t.Fatalf("only %d plans of 256 and 512 points took a 512-bit frame plan", wideTaken)
	}
}
