package fft

import (
	"math"
	"runtime"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The float32 strips (Round 25) against the float32 line-by-line path, on
// stripShapes and the float32 SIMD signals.

// withStrips32 builds a PlanN32 with the strip path forced on or off.
func withStrips32(on bool, shape ...int) *PlanN32 {
	defer func(v bool) { stripAxes32 = v }(stripAxes32)
	stripAxes32 = on
	return f32ndNewPlanN(shape, resolveAxes("test", nil, len(shape)))
}

// f32rAgree is bit identity (NaNs alike whatever their sign), or, where the
// strips may compile to other fusions than the lines (the Go batched pass on
// arm64, any pass under -race on arm64), agreement to 1e-5 of the output's
// scale.
func f32rAgree(a, b complex64, scale float64, exact bool) bool {
	eq := func(x, y float32) bool {
		return math.Float32bits(x) == math.Float32bits(y) || (x != x && y != y)
	}
	if eq(real(a), real(b)) && eq(imag(a), imag(b)) {
		return true
	}
	if exact {
		return false
	}
	near := func(x, y float32) bool {
		if math.IsInf(float64(x), 0) || math.IsInf(float64(y), 0) || x != x || y != y {
			return eq(x, y)
		}
		return math.Abs(float64(x)-float64(y)) <= 1e-5*scale
	}
	return near(real(a), real(b)) && near(imag(a), imag(b))
}

func f32rMaxPart(x []complex64) float64 {
	m := 1.0
	for _, v := range x {
		for _, a := range []float64{math.Abs(float64(real(v))), math.Abs(float64(imag(v)))} {
			if a > m && !math.IsInf(a, 0) {
				m = a
			}
		}
	}
	return m
}

// TestStrips32MatchLines holds the float32 strip path to the line-by-line
// path: forward and inverse, out of place and in place. With the batched
// kernels (the default where they exist) each line of a strip gets the
// arithmetic of its own 1-D kernels, bit for bit outside -race on arm64; the
// Go batched pass (kernels off) is bit-exact on amd64, where gc does not fuse,
// and held to rounding on arm64, where it may fuse otherwise than the lines.
func TestStrips32MatchLines(t *testing.T) {
	defer func(v bool) { kernels.UseStockhamBatch32 = v }(kernels.UseStockhamBatch32)
	modes := []bool{false}
	if kernels.UseStockhamBatch32 {
		modes = append(modes, true)
	}
	// stripShapes, plus twiddled radix-3, -5 and -8 passes over strips of
	// four lines and more (a radix-5 pass with ido > 1 in stripShapes runs on
	// two lines only).
	shapes := append(append([][]int(nil), stripShapes...), []int{25, 16}, []int{125, 7}, []int{9, 20}, []int{27, 13}, []int{64, 21}, []int{8, 4, 9}, []int{40, 6})
	if testing.Short() {
		shapes = shapes[:6]
	}
	for _, kern := range modes {
		kernels.UseStockhamBatch32 = kern
		exact := runtime.GOARCH != "arm64" || (kern && !raceEnabled)
		for _, shape := range shapes {
			lines, strips := withStrips32(false, shape...), withStrips32(true, shape...)
			for ax := range shape {
				want := ax < len(shape)-1 && strips.axes[ax] != nil && f32rStripsFit(strips.axes[ax])
				if (strips.strips[ax] != nil) != want || lines.strips[ax] != nil {
					t.Fatalf("shape %v axis %d: strips %v, want %v", shape, ax, strips.strips[ax] != nil, want)
				}
			}
			n := shapeProduct(shape...)
			for s, x := range f32simdSignals(n) {
				for _, inverse := range []bool{false, true} {
					run := func(p *PlanN32, inPlace bool) []complex64 {
						out := make([]complex64, n)
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
					scale := f32rMaxPart(ref)
					for _, inPlace := range []bool{false, true} {
						got := run(strips, inPlace)
						for i := range ref {
							if !f32rAgree(got[i], ref[i], scale, exact) {
								t.Fatalf("kernels=%v shape %v signal %d inverse=%v in place=%v index %d: strips %v, lines %v",
									kern, shape, s, inverse, inPlace, i, got[i], ref[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestStrips32MatchLinesReal does the same through RealPlan2_32, whose column
// transform is a PlanN32 axis.
func TestStrips32MatchLinesReal(t *testing.T) {
	defer func(v bool) { stripAxes32 = v }(stripAxes32)
	exact := runtime.GOARCH != "arm64" || (kernels.UseStockhamBatch32 && !raceEnabled)
	for _, shape := range [][2]int{{16, 16}, {40, 9}, {6, 3}, {1, 8}, {256, 30}} {
		rows, cols := shape[0], shape[1]
		x := make([]float32, rows*cols)
		for i, v := range cmplxSignal(rows * cols) {
			x[i] = float32(real(v))
		}
		var specs [2][]complex64
		var imgs [2][]float32
		for i, on := range []bool{false, true} {
			stripAxes32 = on
			p := NewRealPlan2_32(rows, cols)
			specs[i] = p.RFFT(make([]complex64, p.SpectrumLen()), x)
			imgs[i] = p.IRFFT(make([]float32, rows*cols), specs[0])
		}
		scale := f32rMaxPart(specs[0])
		for i := range specs[0] {
			if !f32rAgree(specs[1][i], specs[0][i], scale, exact) {
				t.Fatalf("%dx%d RFFT bin %d: strips %v, lines %v", rows, cols, i, specs[1][i], specs[0][i])
			}
		}
		for i := range imgs[0] {
			if !f32rAgree(complex(imgs[1][i], 0), complex(imgs[0][i], 0), scale, exact) {
				t.Fatalf("%dx%d IRFFT index %d: strips %v, lines %v", rows, cols, i, imgs[1][i], imgs[0][i])
			}
		}
	}
}

// TestStrips32AcrossGoroutines runs a shape large enough to fan its strips
// out (parChunks), against the line-by-line path.
func TestStrips32AcrossGoroutines(t *testing.T) {
	defer func(w, m int) { parWorkers, parMinChunk = w, m }(parWorkers, parMinChunk)
	parWorkers, parMinChunk = 4, 64
	shape := []int{256, 128}
	lines, strips := withStrips32(false, shape...), withStrips32(true, shape...)
	x := f32simdSignals(shapeProduct(shape...))[0]
	want := lines.FFT(make([]complex64, len(x)), x)
	got := strips.FFT(make([]complex64, len(x)), x)
	exact := runtime.GOARCH != "arm64" || (kernels.UseStockhamBatch32 && !raceEnabled)
	scale := f32rMaxPart(want)
	for i := range want {
		if !f32rAgree(got[i], want[i], scale, exact) {
			t.Fatalf("index %d: %v vs %v", i, got[i], want[i])
		}
	}
}

// TestBatchTwiddles32Layout pins the float32 batched pass's twiddle order:
// point by point, the r-1 twiddles of each together.
func TestBatchTwiddles32Layout(t *testing.T) {
	p := newSKPlan32Factors(40, []int{8, 5})
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

// TestStrips32Fit: the strip path needs a Stockham plan of two or more
// batched passes.
func TestStrips32Fit(t *testing.T) {
	for _, c := range []struct {
		n    int
		want bool
	}{{64, true}, {40, true}, {8, false}, {14, false}, {1009, false}, {10007, false}} {
		if got := f32rStripsFit(NewPlan32(c.n)); got != c.want {
			t.Errorf("n=%d: fit %v, want %v", c.n, got, c.want)
		}
	}
}
