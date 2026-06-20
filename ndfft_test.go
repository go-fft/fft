package fft

import (
	"math"
	"testing"
)

// naiveDFTN is a reference separable N-dimensional DFT computed by applying the
// naive O(N²) 1-D DFT along each axis in turn. It is the correctness oracle for
// FFTN, independent of the FFT core.
func naiveDFTN(data []complex128, shape []int) []complex128 {
	out := make([]complex128, len(data))
	copy(out, data)
	stride := make([]int, len(shape))
	acc := 1
	for ax := len(shape) - 1; ax >= 0; ax-- {
		stride[ax] = acc
		acc *= shape[ax]
	}
	for ax := range shape {
		n := shape[ax]
		if n == 1 {
			continue
		}
		st := stride[ax]
		lineCount := len(out) / n
		idx := make([]int, len(shape))
		buf := make([]complex128, n)
		for c := 0; c < lineCount; c++ {
			base := 0
			for a, s := range stride {
				base += idx[a] * s
			}
			for i := 0; i < n; i++ {
				buf[i] = out[base+i*st]
			}
			res := naiveDFT(buf)
			for i := 0; i < n; i++ {
				out[base+i*st] = res[i]
			}
			advanceIndex(idx, shape, ax)
		}
	}
	return out
}

func complexGrid(shape []int) []complex128 {
	total := 1
	for _, s := range shape {
		total *= s
	}
	x := make([]complex128, total)
	for i := range x {
		x[i] = complex(math.Sin(float64(i)*0.7)+float64(i%3), math.Cos(float64(i)*0.3)-float64(i%2))
	}
	return x
}

func TestFFTNAgainstNaive(t *testing.T) {
	shapes := [][]int{
		{2, 2}, {3, 4}, {4, 3}, {1, 5}, {5, 1},
		{2, 3, 4}, {3, 1, 2}, {2, 2, 2, 2}, {6, 5},
	}
	for _, shape := range shapes {
		x := complexGrid(shape)
		closeVec(t, FFTN(x, shape), naiveDFTN(x, shape))
	}
}

func TestFFTNRoundTrip(t *testing.T) {
	shapes := [][]int{{4, 4}, {3, 5}, {2, 3, 4}, {7, 2}, {1, 1}}
	for _, shape := range shapes {
		x := complexGrid(shape)
		got := IFFTN(FFTN(x, shape), shape)
		closeVec(t, got, x)
	}
}

func TestFFT2MatchesFFTN(t *testing.T) {
	shape := [2]int{4, 6}
	x := complexGrid(shape[:])
	closeVec(t, FFT2(x, shape), FFTN(x, shape[:]))
	closeVec(t, IFFT2(FFT2(x, shape), shape), x)
}

// TestFFT2Separable confirms a 2-D FFT equals 1-D FFTs along rows then columns.
func TestFFT2Separable(t *testing.T) {
	rows, cols := 3, 4
	shape := [2]int{rows, cols}
	x := complexGrid(shape[:])

	// Manual: FFT each row, then FFT each column.
	tmp := make([]complex128, rows*cols)
	rowbuf := make([]complex128, cols)
	for r := 0; r < rows; r++ {
		copy(rowbuf, x[r*cols:(r+1)*cols])
		rr := FFT(rowbuf)
		copy(tmp[r*cols:(r+1)*cols], rr)
	}
	colbuf := make([]complex128, rows)
	for c := 0; c < cols; c++ {
		for r := 0; r < rows; r++ {
			colbuf[r] = tmp[r*cols+c]
		}
		cc := FFT(colbuf)
		for r := 0; r < rows; r++ {
			tmp[r*cols+c] = cc[r]
		}
	}
	closeVec(t, FFT2(x, shape), tmp)
}

func TestFFTNConstantGivesDCSpike(t *testing.T) {
	shape := []int{3, 4}
	total := 12
	x := make([]complex128, total)
	for i := range x {
		x[i] = 1
	}
	X := FFTN(x, shape)
	want := make([]complex128, total)
	want[0] = complex(float64(total), 0)
	closeVec(t, X, want)
}

func TestFFTNImpulseGivesFlat(t *testing.T) {
	shape := []int{4, 4}
	x := make([]complex128, 16)
	x[0] = 1
	X := FFTN(x, shape)
	want := make([]complex128, 16)
	for i := range want {
		want[i] = 1
	}
	closeVec(t, X, want)
}

func TestFFTNLinearity(t *testing.T) {
	shape := []int{3, 5}
	a, b := complex(2, -1), complex(0.5, 3)
	x := complexGrid(shape)
	y := make([]complex128, len(x))
	for i := range y {
		y[i] = complex(math.Cos(float64(i)), float64(i%4))
	}
	combined := make([]complex128, len(x))
	for i := range combined {
		combined[i] = a*x[i] + b*y[i]
	}
	got := FFTN(combined, shape)
	Fx, Fy := FFTN(x, shape), FFTN(y, shape)
	want := make([]complex128, len(x))
	for i := range want {
		want[i] = a*Fx[i] + b*Fy[i]
	}
	closeVec(t, got, want)
}

func TestFFTNEmptyShape(t *testing.T) {
	// Empty shape, single scalar: identity.
	x := []complex128{complex(3, -2)}
	got := FFTN(x, []int{})
	if len(got) != 1 || got[0] != x[0] {
		t.Fatalf("empty shape: got %v", got)
	}
	got = IFFTN(x, []int{})
	if len(got) != 1 || got[0] != x[0] {
		t.Fatalf("empty shape ifft: got %v", got)
	}
}

func TestFFTNNoMutation(t *testing.T) {
	shape := []int{3, 4}
	x := complexGrid(shape)
	orig := make([]complex128, len(x))
	copy(orig, x)
	FFTN(x, shape)
	closeVec(t, x, orig)
	IFFTN(x, shape)
	closeVec(t, x, orig)
}

func TestFFTNPanicsBadShape(t *testing.T) {
	cases := []struct {
		shape []int
		total int
	}{
		{[]int{3, 4}, 10}, // product mismatch
		{[]int{0, 4}, 0},  // non-positive length (0)
		{[]int{-1, 4}, 4}, // negative length
	}
	for _, c := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("shape %v total %d: expected panic", c.shape, c.total)
				}
			}()
			FFTN(make([]complex128, c.total), c.shape)
		}()
	}
}
