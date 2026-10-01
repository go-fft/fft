package fft

import (
	"math"
	"testing"
)

var realPlan2Shapes = [][2]int{{1, 1}, {1, 8}, {8, 1}, {4, 6}, {5, 7}, {16, 16}, {12, 30}, {30, 12}, {64, 50}, {3, 129}, {300, 70}}

func realMatrix(rows, cols int) []float64 {
	x := make([]float64, rows*cols)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.37) + float64(i%5) - 2
	}
	return x
}

// TestRealPlan2AgainstComplex checks RFFT against the first cols/2+1 columns
// of the complex 2-D FFT of the same real matrix, an oracle that shares no row
// code with it, and that IRFFT inverts RFFT.
func TestRealPlan2AgainstComplex(t *testing.T) {
	for _, s := range realPlan2Shapes {
		rows, cols := s[0], s[1]
		x := realMatrix(rows, cols)
		cx := make([]complex128, len(x))
		for i, v := range x {
			cx[i] = complex(v, 0)
		}
		full := FFT2(cx, s)
		p := NewRealPlan2(rows, cols)
		got := p.RFFT(make([]complex128, p.SpectrumLen()), x)
		rc := cols/2 + 1
		tol := 1e-9 * float64(len(x))
		for r := 0; r < rows; r++ {
			for c := 0; c < rc; c++ {
				if d := cmplxAbs(got[r*rc+c] - full[r*cols+c]); d > tol {
					t.Fatalf("shape %v bin (%d,%d): %v vs complex FFT2 %v", s, r, c, got[r*rc+c], full[r*cols+c])
				}
			}
		}
		back := p.IRFFT(make([]float64, len(x)), got)
		for i := range x {
			if math.Abs(back[i]-x[i]) > 1e-9 {
				t.Fatalf("shape %v: IRFFT(RFFT(x))[%d] = %v, want %v", s, i, back[i], x[i])
			}
		}
	}
}

func cmplxAbs(z complex128) float64 { return math.Hypot(real(z), imag(z)) }

// TestRealPlan2ParallelMatchesSerial forces the worker pool on and off.
func TestRealPlan2ParallelMatchesSerial(t *testing.T) {
	for _, s := range [][2]int{{256, 256}, {300, 70}, {70, 301}} {
		p := NewRealPlan2(s[0], s[1])
		x := realMatrix(s[0], s[1])
		var a, b []complex128
		var ia, ib []float64
		withWorkers(1, func() {
			a = p.RFFT(make([]complex128, p.SpectrumLen()), x)
			ia = p.IRFFT(make([]float64, len(x)), a)
		})
		withWorkers(8, func() {
			b = p.RFFT(make([]complex128, p.SpectrumLen()), x)
			ib = p.IRFFT(make([]float64, len(x)), b)
		})
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("shape %v RFFT bin %d: serial %v parallel %v", s, i, a[i], b[i])
			}
		}
		for i := range ia {
			if ia[i] != ib[i] {
				t.Fatalf("shape %v IRFFT index %d: serial %v parallel %v", s, i, ia[i], ib[i])
			}
		}
	}
}

// TestRealPlan2AllocatesNothing: an even-width plan allocates nothing in
// steady state, forward or inverse.
func TestRealPlan2AllocatesNothing(t *testing.T) {
	if raceEnabled {
		t.Skip("-race makes sync.Pool drop items on purpose; the arch CI jobs run this without it")
	}
	withWorkers(1, func() {
		for _, s := range [][2]int{{64, 64}, {30, 12}, {1, 256}} {
			p := NewRealPlan2(s[0], s[1])
			x := realMatrix(s[0], s[1])
			spec := make([]complex128, p.SpectrumLen())
			out := make([]float64, len(x))
			p.RFFT(spec, x)
			p.IRFFT(out, spec)
			if a := testing.AllocsPerRun(20, func() { p.RFFT(spec, x) }); a != 0 {
				t.Errorf("shape %v: RFFT allocates %v times per call", s, a)
			}
			if a := testing.AllocsPerRun(20, func() { p.IRFFT(out, spec) }); a != 0 {
				t.Errorf("shape %v: IRFFT allocates %v times per call", s, a)
			}
		}
	})
}

func TestRealPlan2Panics(t *testing.T) {
	expectPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s did not panic", name)
			}
		}()
		f()
	}
	expectPanic("zero rows", func() { NewRealPlan2(0, 4) })
	expectPanic("zero cols", func() { NewRealPlan2(4, 0) })
	p := NewRealPlan2(4, 6)
	expectPanic("RFFT short src", func() { p.RFFT(make([]complex128, 16), make([]float64, 23)) })
	expectPanic("RFFT short dst", func() { p.RFFT(make([]complex128, 15), make([]float64, 24)) })
	expectPanic("IRFFT short dst", func() { p.IRFFT(make([]float64, 23), make([]complex128, 16)) })
	expectPanic("IRFFT short src", func() { p.IRFFT(make([]float64, 24), make([]complex128, 15)) })
}
