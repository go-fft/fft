package fft

import (
	"math"
	"testing"
)

// psdRef is the per-bin formula of the PSD before the power pass became a
// go-simd/floats kernel: (re²+im²)·scale, the interior bins doubled.
func psdRef(out []float64, spec []complex128, n int, d float64) {
	m := len(spec)
	scale := d / float64(n)
	for k := 0; k < m; k++ {
		re, im := real(spec[k]), imag(spec[k])
		p := (re*re + im*im) * scale
		if k != 0 && !(n%2 == 0 && k == m-1) {
			p *= 2
		}
		out[k] = p
	}
}

// TestPSDKernelRuns: the kernel's three runs (DC, the doubled interior, the Nyquist
// bin of an even length) cover every bin exactly once and agree with the formula to
// the kernel's bound (4 ulp: the kernel fuses the second square into the sum), for
// odd and even lengths, from 1 to 2,049 samples.
func TestPSDKernelRuns(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100, 255, 256, 257, 1000, 2048, 2049} {
		m := n/2 + 1
		spec := make([]complex128, m)
		x := accRealF64(2*m, 95)
		for k := range spec {
			spec[k] = complex(x[2*k]*10, x[2*k+1]*10)
		}
		got, want := make([]float64, m), make([]float64, m)
		for i := range got {
			got[i] = math.NaN() // a bin the runs skipped would stay NaN
		}
		psdFromSpectrum(got, spec, n, 0.001)
		psdRef(want, spec, n, 0.001)
		for k := range got {
			if math.Abs(got[k]-want[k]) > 4*0x1p-53*math.Abs(want[k]) {
				t.Fatalf("n=%d bin %d: %g, want %g", n, k, got[k], want[k])
			}
		}
	}
}

// TestRealPackedWindowed: the window multiplied inside the packing gather is the
// transform of the product, bit for bit, for the iterative kernel directly and for a
// RealPlan whose half runs on it (on architectures whose routing does not pick it for
// these lengths the plan is given one, so the path is exercised everywhere).
func TestRealPackedWindowed(t *testing.T) {
	isolatePlans(t)
	for _, n := range []int{2, 4, 8, 64, 128, 1024} { // n: the half-length complex plan
		src, win := accRealF64(2*n, 97), accRealF64(2*n, 98)
		prod := make([]float64, 2*n)
		for i := range prod {
			prod[i] = src[i] * win[i]
		}
		it := newITPlan(n)
		got, want := make([]complex128, n), make([]complex128, n)
		it.transformRealPackedWindowed(got, src, win)
		it.transformRealPacked(want, prod)
		for k := range got {
			if got[k] != want[k] {
				t.Fatalf("itPlan n=%d: bin %d differs", n, k)
			}
		}
		plan := NewRealPlan(2 * n)
		if plan.half == nil {
			continue // 2n = 2: no half-length plan
		}
		if plan.half.it == nil {
			plan.half.it = it
		}
		if !plan.windowedPacked() {
			t.Fatalf("n=%d: the plan does not report the windowed path", 2*n)
		}
		g, w := make([]complex128, n+1), make([]complex128, n+1)
		plan.rfftWindowed(g, src, win)
		plan.RFFT(w, prod)
		for k := range g {
			if g[k] != w[k] {
				t.Fatalf("RealPlan 2n=%d: bin %d differs from RFFT of the product", 2*n, k)
			}
		}
	}
}

// TestSpectrogramWindowPaths: a power-of-two segment takes the fused path (the
// window multiplied inside the real transform's packing gather), any other the
// windowed copy; each frame is PSD of the windowed segment bit for bit, either way.
func TestSpectrogramWindowPaths(t *testing.T) {
	isolatePlans(t)
	x := accRealF64(5000, 96)
	for _, segment := range []int{2, 3, 8, 100, 128, 255, 256, 512} {
		if segment >= 8 && segment&(segment-1) == 0 {
			// Give the cached plan the iterative half, so the fused path runs on
			// every architecture (a no-op where routing already chose it).
			if p := cachedRealPlan(segment); p.half != nil && p.half.it == nil {
				p.half.it = newITPlan(segment / 2)
			}
		}
		w := Hann(segment)
		for _, overlap := range []int{0, segment / 2, segment - 1} {
			frames := Spectrogram(x, segment, overlap, w, 0.01)
			step := segment - overlap
			for f, got := range frames {
				seg := make([]float64, segment)
				for i := range seg {
					seg[i] = x[f*step+i] * w[i]
				}
				want := PSD(seg, 0.01)
				for k := range want {
					if got[k] != want[k] {
						t.Fatalf("segment=%d overlap=%d frame %d bin %d: %g, want %g", segment, overlap, f, k, got[k], want[k])
					}
				}
			}
		}
	}
}
