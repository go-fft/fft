package fft

import "testing"

// The twiddle passes are go-simd/floats kernels, which refuse a dst that overlaps an
// operand. go-fft's documented in-place forms (dst aliasing src) must never hand
// them aliased slices: the kernels see the plan's scratch on one side. These tests
// run every in-place form at sizes that take whole kernel steps and require the
// result of the out-of-place call, bit for bit.

func TestInPlaceDCT4NeverReachesKernelsAliased(t *testing.T) {
	for _, n := range []int{2, 4, 6, 8, 16, 62, 64, 128, 1024, 2050} {
		p := NewDCTPlan(n, 4)
		x := accRealF64(n, 71)
		for _, norm := range []Norm{NormBackward, NormOrtho, NormForward} {
			want := p.DCT(make([]float64, n), x, norm)
			buf := append([]float64(nil), x...)
			if got := p.DCT(buf, buf, norm); !equalF(got, want) {
				t.Fatalf("n=%d %v: in-place DCT differs", n, norm)
			}
			wantI := p.IDCT(make([]float64, n), x, norm)
			buf = append([]float64(nil), x...)
			if got := p.IDCT(buf, buf, norm); !equalF(got, wantI) {
				t.Fatalf("n=%d %v: in-place IDCT differs", n, norm)
			}
			x32 := accRealF32(n, 72)
			want32 := p.DCTMixed(make([]float32, n), x32, norm)
			b32 := append([]float32(nil), x32...)
			got32 := p.DCTMixed(b32, b32, norm)
			for i := range got32 {
				if got32[i] != want32[i] {
					t.Fatalf("n=%d %v: in-place DCTMixed differs at %d", n, norm, i)
				}
			}
		}
	}
}

func TestInPlaceMDCTNeverReachesKernelsAliased(t *testing.T) {
	for _, n := range []int{4, 8, 16, 128, 256, 2048} {
		m := n / 2
		p := NewMDCTPlan(n)
		x, c := accRealF64(n, 73), accRealF64(m, 74)
		// MDCT: dst overlaps the start of src
		want := p.MDCT(make([]float64, m), x, 0.5)
		buf := append([]float64(nil), x...)
		if got := p.MDCT(buf[:m], buf, 0.5); !equalF(got, want) {
			t.Fatalf("N=%d: MDCT into the start of its src differs", n)
		}
		// IMDCT: dst is src's first half's neighbour, then exactly src extended
		wantI := p.IMDCT(make([]float64, n), c, 0.5)
		buf = append(append([]float64(nil), c...), make([]float64, n-m)...)
		if got := p.IMDCT(buf, buf, 0.5); !equalF(got, wantI) {
			t.Fatalf("N=%d: IMDCT over its src differs", n)
		}
		x32, c32 := accRealF32(n, 75), accRealF32(m, 76)
		want32 := p.IMDCTMixed(make([]float32, n), c32, 0.5)
		b32 := append(append([]float32(nil), c32...), make([]float32, n-m)...)
		got32 := p.IMDCTMixed(b32, b32, 0.5)
		for i := range got32 {
			if got32[i] != want32[i] {
				t.Fatalf("N=%d: IMDCTMixed over its src differs at %d", n, i)
			}
		}
		wantM := p.MDCTMixed(make([]float32, m), x32, 0.5)
		bx := append([]float32(nil), x32...)
		gotM := p.MDCTMixed(bx[:m], bx, 0.5)
		for i := range gotM {
			if gotM[i] != wantM[i] {
				t.Fatalf("N=%d: MDCTMixed into the start of its src differs at %d", n, i)
			}
		}
	}
}
