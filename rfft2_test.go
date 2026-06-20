package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

func realGrid(shape [2]int) []float64 {
	x := make([]float64, shape[0]*shape[1])
	for i := range x {
		x[i] = math.Sin(float64(i)*0.7) + 0.3*math.Cos(float64(i)*1.9) + float64(i%4)
	}
	return x
}

// TestRFFT2MatchesFFT2 cross-checks RFFT2 against the kept bins of the full
// complex 2-D transform: numpy.fft.rfft2(a)[r, c] == fft2(a)[r, c] for the kept
// last-axis bins c in [0, cols/2].
func TestRFFT2MatchesFFT2(t *testing.T) {
	for _, shape := range [][2]int{{4, 6}, {3, 4}, {5, 8}, {2, 2}, {4, 5}, {1, 6}, {6, 1}} {
		rows, cols := shape[0], shape[1]
		rcols := cols/2 + 1
		x := realGrid(shape)

		full := FFT2(toComplex(x), shape)
		got := RFFT2(x, shape)
		if len(got) != rows*rcols {
			t.Fatalf("shape %v: len %d, want %d", shape, len(got), rows*rcols)
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < rcols; c++ {
				want := full[r*cols+c]
				g := got[r*rcols+c]
				if d := cmplx.Abs(g - want); d > tol {
					t.Fatalf("shape %v bin (%d,%d): got %v want %v |d|=%g", shape, r, c, g, want, d)
				}
			}
		}
	}
}

func toComplex(x []float64) []complex128 {
	c := make([]complex128, len(x))
	for i, v := range x {
		c[i] = complex(v, 0)
	}
	return c
}

func TestRFFT2RoundTrip(t *testing.T) {
	for _, shape := range [][2]int{{4, 6}, {3, 5}, {2, 8}, {5, 4}, {1, 1}, {7, 3}} {
		x := realGrid(shape)
		got := IRFFT2(RFFT2(x, shape), shape)
		if len(got) != shape[0]*shape[1] {
			t.Fatalf("shape %v: len %d", shape, len(got))
		}
		for i := range got {
			if d := math.Abs(got[i] - x[i]); d > tol {
				t.Fatalf("shape %v index %d: got %g want %g |d|=%g", shape, i, got[i], x[i], d)
			}
		}
	}
}

func TestRFFT2ConstantDC(t *testing.T) {
	shape := [2]int{3, 4}
	x := make([]float64, 12)
	for i := range x {
		x[i] = 1
	}
	X := RFFT2(x, shape)
	rcols := shape[1]/2 + 1
	for i, v := range X {
		want := 0.0
		if i == 0 {
			want = 12
		}
		if d := cmplx.Abs(v - complex(want, 0)); d > tol {
			t.Fatalf("bin %d (rcols=%d): got %v want %g", i, rcols, v, want)
		}
	}
}

func TestIRFFT2ShortSpectrumZeroPads(t *testing.T) {
	// A spectrum shorter than rows*(cols/2+1) must be treated as zero-padded.
	shape := [2]int{2, 4}
	full := RFFT2(realGrid(shape), shape)
	short := full[:len(full)-1]
	got := IRFFT2(short, shape)

	padded := make([]complex128, len(full))
	copy(padded, short)
	want := IRFFT2(padded, shape)
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
		}
	}
}

func TestRFFT2NoMutation(t *testing.T) {
	shape := [2]int{3, 4}
	x := realGrid(shape)
	orig := make([]float64, len(x))
	copy(orig, x)
	RFFT2(x, shape)
	for i := range x {
		if x[i] != orig[i] {
			t.Fatalf("RFFT2 mutated input at %d", i)
		}
	}

	spec := RFFT2(x, shape)
	specOrig := make([]complex128, len(spec))
	copy(specOrig, spec)
	IRFFT2(spec, shape)
	for i := range spec {
		if spec[i] != specOrig[i] {
			t.Fatalf("IRFFT2 mutated input at %d", i)
		}
	}
}

func TestRFFT2PanicsBadShape(t *testing.T) {
	cases := []struct {
		fn    func()
		label string
	}{
		{func() { RFFT2(make([]float64, 4), [2]int{0, 4}) }, "rfft2 zero dim"},
		{func() { RFFT2(make([]float64, 4), [2]int{2, -4}) }, "rfft2 neg dim"},
		{func() { RFFT2(make([]float64, 5), [2]int{2, 4}) }, "rfft2 product mismatch"},
		{func() { IRFFT2(make([]complex128, 4), [2]int{0, 4}) }, "irfft2 zero dim"},
		{func() { IRFFT2(make([]complex128, 4), [2]int{2, -4}) }, "irfft2 neg dim"},
	}
	for _, c := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s: expected panic", c.label)
				}
			}()
			c.fn()
		}()
	}
}
