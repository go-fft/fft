package fft

import (
	"math"
	"testing"
)

// stripShapes are the N-D shapes the strip tests run: every batched radix
// (2, 3, 4, 5, 8, and 16 on amd64 with AVX2) on a strided axis, strips narrower than stripWidth, an odd
// number of lines (the 128-bit tail of the kernels), single-column axes, and
// axes that keep the gather path (radix 7, a single pass, Bluestein).
var stripShapes = [][]int{
	{6, 4}, {16, 8}, {40, 5}, {64, 64}, {128, 128}, {12, 1}, {45, 3}, {1000, 2},
	{4, 6, 5}, {10, 3, 7}, {2, 16, 9}, {48, 1, 6}, {3, 4},
	{14, 6}, {8, 8}, {17, 4}, {1, 9}, {16, 40}, {6, 33}, {1024, 35},
	{256, 3}, {128, 5}, {2048, 3}, // radix 16 where radix16Table has it
}

// withStrips builds a plan with the strip path forced on or off.
func withStrips(on bool, shape ...int) *PlanN {
	defer func(v bool) { stripAxes = v }(stripAxes)
	stripAxes = on
	return NewPlanN(shape...)
}

// agree is bit identity, or, where the scalar passes fuse multiply-adds
// (scalarFuses), agreement of each part to a few ulps of the output's scale
// when finite, and bit identity when not.
func agree(a, b complex128, scale float64) bool {
	if sameBits(a, b) {
		return true
	}
	if !scalarFuses {
		return false
	}
	near := func(x, y float64) bool {
		if math.IsInf(x, 0) || math.IsInf(y, 0) || math.IsNaN(x) || math.IsNaN(y) {
			return sameBits(complex(x, 0), complex(y, 0))
		}
		return math.Abs(x-y) <= 1e-12*scale
	}
	return near(real(a), real(b)) && near(imag(a), imag(b))
}

// maxPart is the largest finite magnitude of a real or imaginary part in x,
// at least 1.
func maxPart(x []complex128) float64 {
	m := 1.0
	for _, v := range x {
		for _, a := range []float64{math.Abs(real(v)), math.Abs(imag(v))} {
			if a > m && !math.IsInf(a, 0) {
				m = a
			}
		}
	}
	return m
}

// TestStripsMatchLines holds the batched (strip) path to the line-by-line
// path, bit for bit: forward and inverse, out of place and in place, on the
// generic, signed-zero and infinite signals. Each line of a strip gets its own
// 1-D transform's arithmetic, so nothing may differ.
func TestStripsMatchLines(t *testing.T) {
	for _, shape := range stripShapes {
		lines, strips := withStrips(false, shape...), withStrips(true, shape...)
		for ax := range shape[:len(shape)-1] {
			want := ax < len(shape)-1 && strips.axes[ax] != nil && stripsFit(strips.axes[ax])
			if (strips.strips[ax] != nil) != want || lines.strips[ax] != nil {
				t.Fatalf("shape %v axis %d: strips %v, want %v", shape, ax, strips.strips[ax] != nil, want)
			}
		}
		n := shapeProduct(shape...)
		for s, x := range simdSignals(n) {
			for _, inverse := range []bool{false, true} {
				run := func(p *PlanN, inPlace bool) []complex128 {
					out := make([]complex128, n)
					src := x
					if inPlace {
						copy(out, x)
						src = out
					}
					if inverse {
						return p.IFFT(out, src)
					}
					return p.FFT(out, src)
				}
				ref := run(lines, false)
				scale := maxPart(ref)
				for _, inPlace := range []bool{false, true} {
					got := run(strips, inPlace)
					for i := range ref {
						if !agree(got[i], ref[i], scale) {
							t.Fatalf("shape %v signal %d inverse=%v in place=%v index %d: strips %v, lines %v",
								shape, s, inverse, inPlace, i, got[i], ref[i])
						}
					}
				}
			}
		}
	}
}

// TestStripsMatchLinesReal does the same through RealPlan2, whose column
// transform is a PlanN axis.
func TestStripsMatchLinesReal(t *testing.T) {
	defer func(v bool) { stripAxes = v }(stripAxes)
	for _, shape := range [][2]int{{16, 16}, {40, 9}, {6, 3}, {1, 8}} {
		rows, cols := shape[0], shape[1]
		x := make([]float64, rows*cols)
		for i, v := range cmplxSignal(rows * cols) {
			x[i] = real(v)
		}
		var specs [2][]complex128
		var imgs [2][]float64
		for i, on := range []bool{false, true} {
			stripAxes = on
			p := NewRealPlan2(rows, cols)
			specs[i] = p.RFFT(make([]complex128, p.SpectrumLen()), x)
			imgs[i] = p.IRFFT(make([]float64, rows*cols), specs[i])
		}
		scale := maxPart(specs[0])
		for i := range specs[0] {
			if !agree(specs[0][i], specs[1][i], scale) {
				t.Fatalf("%dx%d RFFT bin %d: strips %v, lines %v", rows, cols, i, specs[1][i], specs[0][i])
			}
		}
		for i := range imgs[0] {
			if !agree(complex(imgs[0][i], 0), complex(imgs[1][i], 0), scale) {
				t.Fatalf("%dx%d IRFFT index %d: strips %v, lines %v", rows, cols, i, imgs[1][i], imgs[0][i])
			}
		}
	}
}

// TestStripsAcrossGoroutines runs a shape large enough to fan its strips out
// (parChunks), and checks it against the line-by-line path.
func TestStripsAcrossGoroutines(t *testing.T) {
	defer func(w, m int) { parWorkers, parMinChunk = w, m }(parWorkers, parMinChunk)
	parWorkers, parMinChunk = 4, 64
	shape := []int{256, 128}
	lines, strips := withStrips(false, shape...), withStrips(true, shape...)
	x := cmplxSignal(shapeProduct(shape...))
	want := lines.FFT(make([]complex128, len(x)), x)
	got := strips.FFT(make([]complex128, len(x)), x)
	scale := maxPart(want)
	for i := range want {
		if !agree(got[i], want[i], scale) {
			t.Fatalf("index %d: %v vs %v", i, got[i], want[i])
		}
	}
}

// TestBatchTwiddlesLayout pins the batched pass's twiddle order: point by
// point, the r-1 twiddles of each together.
func TestBatchTwiddlesLayout(t *testing.T) {
	p := newSKPlan(40) // radix 8 then 5
	for _, st := range p.stages {
		fwd, conj := st.batchTwiddles()
		if st.ido == 1 {
			if fwd != nil || conj != nil {
				t.Fatalf("ido 1: want no table")
			}
			continue
		}
		for i := 1; i < st.ido; i++ {
			for j := 1; j < st.r; j++ {
				w := st.tw[(j-1)*(st.ido-1)+i-1]
				if got := fwd[(i-1)*(st.r-1)+j-1]; got != w {
					t.Fatalf("r=%d i=%d j=%d: %v, want %v", st.r, i, j, got, w)
				}
				if got := conj[(i-1)*(st.r-1)+j-1]; got != complex(real(w), -imag(w)) {
					t.Fatalf("conjugate r=%d i=%d j=%d: %v", st.r, i, j, got)
				}
			}
		}
	}
}
