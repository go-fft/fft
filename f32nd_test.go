package fft

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"testing"
)

// The single-precision N-D, Options and DCT/DST entry points are held to the
// same accuracy as the 1-D ones (bound32: eps32·log2(n), relative 2-norm, n
// the number of points one output depends on) against the float64 library
// path on the same input widened exactly, which the float64 tests check
// against the definitions.

// f32ndWorst records the largest error seen, as a fraction of the bound.
type f32ndWorst struct {
	mu   sync.Mutex
	frac float64
	what string
}

// check fails t if got is further than bound from want and records the
// fraction of the bound used.
func (w *f32ndWorst) check(t *testing.T, what string, e, bound float64) {
	t.Helper()
	f := e / bound
	if !(f <= 1) {
		t.Errorf("%s: error %.3g is %.3g× the bound", what, e, f)
		return
	}
	w.mu.Lock()
	if f > w.frac {
		w.frac, w.what = f, what
	}
	w.mu.Unlock()
}

func (w *f32ndWorst) log(t *testing.T) {
	t.Helper()
	t.Logf("worst error %.3f of eps32·log2(n) (%s)", w.frac, w.what)
}

// real32 returns n random samples in [-1,1) and the same values widened.
func real32(n int, seed uint64) ([]float32, []float64) {
	r := rand.New(rand.NewPCG(seed, uint64(n)+7))
	x := make([]float32, n)
	w := make([]float64, n)
	for i := range x {
		x[i] = float32(r.Float64()*2 - 1)
		w[i] = float64(x[i])
	}
	return x, w
}

// f32ndShapes are the N-D shapes tested: scalar, 1-D, length-1 axes in every
// position, smooth, prime (Rader) and Bluestein lengths, and 2-D shapes past
// the parallel threshold (with a non-contiguous axis long enough to narrow
// the gather block).
var f32ndShapes = [][]int{
	{}, {1}, {6}, {1, 1}, {4, 4}, {1, 8}, {8, 1}, {3, 5, 7}, {16, 1, 9},
	{2, 3, 4, 5}, {17, 23}, {12, 10}, {33, 17, 2}, {128, 130}, {4096, 5},
}

func f32ndProduct(shape []int) int {
	p := 1
	for _, s := range shape {
		p *= s
	}
	return p
}

// f32ndAxesChoices are the Options.Axes tried on a shape of ndim axes.
func f32ndAxesChoices(ndim int) [][]int {
	out := [][]int{nil}
	if ndim >= 1 {
		out = append(out, []int{-1}, []int{0})
	}
	if ndim >= 2 {
		out = append(out, []int{1, 0}, []int{0, -1})
	}
	if ndim >= 3 {
		out = append(out, []int{2, 0})
	}
	return out
}

// widen returns x as complex128.
func widen(x []complex64) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = complex128(v)
	}
	return out
}

func widenReal(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}
	return out
}

// round64 rounds x to complex64.
func round64(x []complex128) []complex64 { return toC64(x) }

// axesPoints is the number of points a transform over the listed axes
// combines: the product of their lengths (at least 1).
func axesPoints(shape, axes []int) int {
	if axes == nil {
		return f32ndProduct(shape)
	}
	p := 1
	for _, ax := range axes {
		if ax < 0 {
			ax += len(shape)
		}
		p *= shape[ax]
	}
	return p
}

// TestFFTN32AgainstFloat64 holds FFTN32With/IFFTN32With (every axis choice
// and Norm), FFTN32/IFFTN32, FFT2_32/IFFT2_32 and PlanN32 to bound32 against
// the float64 N-D transforms, serially and across goroutines.
func TestFFTN32AgainstFloat64(t *testing.T) {
	var w f32ndWorst
	for _, workers := range []int{1, 4} {
		withWorkers(workers, func() {
			for si, shape := range f32ndShapes {
				total := f32ndProduct(shape)
				x, x64 := signal32(total, uint64(si))
				for _, axes := range f32ndAxesChoices(len(shape)) {
					b := bound32(axesPoints(shape, axes))
					for _, m := range allNorms {
						o := Options{Axes: axes, Norm: m}
						what := fmt.Sprintf("shape %v axes %v %v workers %d", shape, axes, m, workers)
						w.check(t, what+" FFTN32With", relErr(FFTN32With(x, shape, o), FFTNWith(x64, shape, o)), b)
						w.check(t, what+" IFFTN32With", relErr(IFFTN32With(x, shape, o), IFFTNWith(x64, shape, o)), b)
						if len(shape) == 2 {
							s2 := [2]int{shape[0], shape[1]}
							w.check(t, what+" FFT2_32With", relErr(FFT2_32With(x, s2, o), FFT2With(x64, s2, o)), b)
							w.check(t, what+" IFFT2_32With", relErr(IFFT2_32With(x, s2, o), IFFT2With(x64, s2, o)), b)
						}
					}
				}
				b := bound32(total)
				w.check(t, fmt.Sprint(shape, " FFTN32"), relErr(FFTN32(x, shape), FFTN(x64, shape)), b)
				w.check(t, fmt.Sprint(shape, " IFFTN32"), relErr(IFFTN32(x, shape), IFFTN(x64, shape)), b)
				if len(shape) == 2 {
					s2 := [2]int{shape[0], shape[1]}
					w.check(t, fmt.Sprint(shape, " FFT2_32"), relErr(FFT2_32(x, s2), FFT2(x64, s2)), b)
					w.check(t, fmt.Sprint(shape, " IFFT2_32"), relErr(IFFT2_32(x, s2), IFFT2(x64, s2)), b)
				}
				p := NewPlanN32(shape...)
				if p.Len() != total || !slices.Equal(p.Shape(), shape) {
					t.Fatalf("NewPlanN32(%v): Len %d Shape %v", shape, p.Len(), p.Shape())
				}
				dst := make([]complex64, total)
				w.check(t, fmt.Sprint(shape, " PlanN32.FFT"), relErr(p.FFT(dst, x), FFTN(x64, shape)), b)
				w.check(t, fmt.Sprint(shape, " PlanN32.IFFT"), relErr(p.IFFT(dst, x), IFFTN(x64, shape)), b)
				for _, m := range allNorms {
					o := Options{Norm: m}
					w.check(t, fmt.Sprint(shape, m, " PlanN32.FFTNorm"), relErr(p.FFTNorm(dst, x, m), FFTNWith(x64, shape, o)), b)
					// In place.
					copy(dst, x)
					w.check(t, fmt.Sprint(shape, m, " PlanN32.IFFTNorm in place"), relErr(p.IFFTNorm(dst, dst, m), IFFTNWith(x64, shape, o)), b)
				}
			}
		})
	}
	w.log(t)
}

// TestRFFTN32AgainstFloat64 holds the real N-D pair, the real 2-D functions
// and RealPlan2_32 to bound32 against their float64 counterparts.
func TestRFFTN32AgainstFloat64(t *testing.T) {
	var w f32ndWorst
	for _, workers := range []int{1, 4} {
		withWorkers(workers, func() {
			for si, shape := range f32ndShapes {
				if len(shape) == 0 {
					continue // RFFTN needs an axis
				}
				total := f32ndProduct(shape)
				x, x64 := real32(total, uint64(si))
				for _, axes := range f32ndAxesChoices(len(shape)) {
					b := bound32(axesPoints(shape, axes))
					for _, m := range allNorms {
						o := Options{Axes: axes, Norm: m}
						what := fmt.Sprintf("shape %v axes %v %v workers %d", shape, axes, m, workers)
						S64 := RFFTNWith(x64, shape, o)
						w.check(t, what+" RFFTN32With", relErr(RFFTN32With(x, shape, o), S64), b)
						S := round64(S64)
						w.check(t, what+" IRFFTN32With", relErrReal(IRFFTN32With(S, shape, o), IRFFTNWith(widen(S), shape, o)), b)
						if len(shape) == 2 {
							s2 := [2]int{shape[0], shape[1]}
							w.check(t, what+" RFFT2_32With", relErr(RFFT2_32With(x, s2, o), RFFT2With(x64, s2, o)), b)
							w.check(t, what+" IRFFT2_32With", relErrReal(IRFFT2_32With(S, s2, o), IRFFT2With(widen(S), s2, o)), b)
						}
					}
				}
				b := bound32(total)
				S64 := RFFTN(x64, shape)
				S := round64(S64)
				w.check(t, fmt.Sprint(shape, " RFFTN32"), relErr(RFFTN32(x, shape), S64), b)
				w.check(t, fmt.Sprint(shape, " IRFFTN32"), relErrReal(IRFFTN32(S, shape), IRFFTN(widen(S), shape)), b)
				if len(shape) != 2 {
					continue
				}
				s2 := [2]int{shape[0], shape[1]}
				w.check(t, fmt.Sprint(shape, " RFFT2_32"), relErr(RFFT2_32(x, s2), RFFT2(x64, s2)), b)
				w.check(t, fmt.Sprint(shape, " IRFFT2_32"), relErrReal(IRFFT2_32(S, s2), IRFFT2(widen(S), s2)), b)
				p := NewRealPlan2_32(shape[0], shape[1])
				spec := make([]complex64, p.SpectrumLen())
				img := make([]float32, total)
				w.check(t, fmt.Sprint(shape, " RealPlan2_32.RFFT"), relErr(p.RFFT(spec, x), RFFT2(x64, s2)), b)
				w.check(t, fmt.Sprint(shape, " RealPlan2_32.IRFFT"), relErrReal(p.IRFFT(img, S), IRFFT2(widen(S), s2)), b)
				for _, m := range allNorms {
					o := Options{Norm: m}
					w.check(t, fmt.Sprint(shape, m, " RealPlan2_32.RFFTNorm"), relErr(p.RFFTNorm(spec, x, m), RFFT2With(x64, s2, o)), b)
					w.check(t, fmt.Sprint(shape, m, " RealPlan2_32.IRFFTNorm"), relErrReal(p.IRFFTNorm(img, S, m), IRFFT2With(widen(S), s2, o)), b)
				}
			}
		})
	}
	w.log(t)
}

// TestIRFFT32NDSpectrumLength: a spectrum shorter than the layout reads as
// zero-padded, a longer one is trimmed, as for IRFFT2 and IRFFTN.
func TestIRFFT32NDSpectrumLength(t *testing.T) {
	shape := []int{4, 6}
	x, _ := real32(24, 3)
	S := RFFTN32(x, shape)
	for _, cut := range []int{len(S) - 5, len(S) + 3} {
		spec := make([]complex64, cut)
		copy(spec, S)
		want := IRFFTN(widen(spec), shape)
		if e := relErrReal(IRFFTN32(spec, shape), want); e > bound32(24) {
			t.Errorf("IRFFTN32 with %d bins: error %g", cut, e)
		}
		if e := relErrReal(IRFFT2_32(spec, [2]int{4, 6}), want); e > bound32(24) {
			t.Errorf("IRFFT2_32 with %d bins: error %g", cut, e)
		}
	}
}

// TestOptions32_1D holds the 1-D ...With functions and the Hermitian pair to
// bound32 against their float64 counterparts, for N cropping, padding and
// default, under every Norm.
func TestOptions32_1D(t *testing.T) {
	var w f32ndWorst
	for _, l := range []int{0, 1, 2, 7, 16, 17, 30, 97, 1000} {
		x, x64 := signal32(l, 5)
		r, r64 := real32(l, 6)
		for _, nn := range []int{0, 1, l / 2, l + 3, 2 * l} {
			for _, m := range allNorms {
				o := Options{N: nn, Norm: m}
				n := nn
				if n == 0 {
					n = l
				}
				b := bound32(n)
				what := fmt.Sprintf("len %d N %d %v", l, nn, m)
				w.check(t, what+" FFT32With", relErr(FFT32With(x, o), FFTWith(x64, o)), b)
				w.check(t, what+" IFFT32With", relErr(IFFT32With(x, o), IFFTWith(x64, o)), b)
				w.check(t, what+" RFFT32With", relErr(RFFT32With(r, o), RFFTWith(r64, o)), b)
				w.check(t, what+" IHFFT32With", relErr(IHFFT32With(r, o), IHFFTWith(r64, o)), b)
				// The spectrum-side functions read x as a half spectrum.
				hn := nn
				if hn == 0 {
					hn = 2 * (l - 1)
				}
				hb := bound32(max(hn, 2))
				w.check(t, what+" IRFFT32With", relErrReal(IRFFT32With(x, o), IRFFTWith(x64, o)), hb)
				w.check(t, what+" HFFT32With", relErrReal(HFFT32With(x, o), HFFTWith(x64, o)), hb)
			}
		}
		w.check(t, fmt.Sprint(l, " IHFFT32"), relErr(IHFFT32(r), IHFFT(r64)), bound32(l))
		for _, n := range []int{-1, 0, 1, l, 2*l + 1} {
			w.check(t, fmt.Sprint(l, " HFFT32 n=", n), relErrReal(HFFT32(x, n), HFFT(x64, n)), bound32(max(n, 2)))
		}
	}
	w.log(t)
}

// f32ndR2R runs the single- or double-precision 1-D DCT/DST named by its
// arguments.
func f32ndR2R(x []float32, cosine bool, typ int, m Norm, inverse bool) []float32 {
	switch {
	case cosine && inverse:
		return IDCT32(x, typ, m)
	case cosine:
		return DCT32(x, typ, m)
	case inverse:
		return IDST32(x, typ, m)
	}
	return DST32(x, typ, m)
}

func f64R2R(x []float64, cosine bool, typ int, m Norm, inverse bool) []float64 {
	switch {
	case cosine && inverse:
		return IDCT(x, typ, m)
	case cosine:
		return DCT(x, typ, m)
	case inverse:
		return IDST(x, typ, m)
	}
	return DST(x, typ, m)
}

// TestR2R32AgainstFloat64 holds every DCT/DST type, direction and Norm, the
// 1-D functions, the plans (also in place) and the N-D forms, to bound32 of
// the logical size against the float64 transforms.
func TestR2R32AgainstFloat64(t *testing.T) {
	var w f32ndWorst
	lengths := []int{1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 64, 97, 100, 127, 128, 255, 256, 1000, 1024, 4096, 4097}
	for _, cosine := range []bool{true, false} {
		for typ := 1; typ <= 4; typ++ {
			for _, n := range lengths {
				if n < minR2RLen(cosine, typ) {
					continue
				}
				x, x64 := real32(n, uint64(typ))
				b := bound32(logicalSize(cosine, typ, n))
				var dp *DCTPlan32
				var sp *DSTPlan32
				if cosine {
					dp = NewDCTPlan32(n, typ)
					if dp.Len() != n || dp.Type() != typ {
						t.Fatalf("NewDCTPlan32(%d, %d): Len %d Type %d", n, typ, dp.Len(), dp.Type())
					}
				} else {
					sp = NewDSTPlan32(n, typ)
					if sp.Len() != n || sp.Type() != typ {
						t.Fatalf("NewDSTPlan32(%d, %d): Len %d Type %d", n, typ, sp.Len(), sp.Type())
					}
				}
				for _, m := range allNorms {
					for _, inverse := range []bool{false, true} {
						what := r2rLabel(cosine, typ, m, inverse, n)
						want := f64R2R(x64, cosine, typ, m, inverse)
						w.check(t, what, relErrReal(f32ndR2R(x, cosine, typ, m, inverse), want), b)
						dst := append([]float32(nil), x...)
						var got []float32
						switch {
						case cosine && inverse:
							got = dp.IDCT(dst, dst, m)
						case cosine:
							got = dp.DCT(dst, dst, m)
						case inverse:
							got = sp.IDST(dst, dst, m)
						default:
							got = sp.DST(dst, dst, m)
						}
						w.check(t, what+" plan in place", relErrReal(got, want), b)
					}
				}
			}
		}
	}
	for _, shape := range [][]int{{}, {5}, {3, 4, 5}, {2, 8}, {16, 17}} {
		total := f32ndProduct(shape)
		x, x64 := real32(total, 9)
		for _, cosine := range []bool{true, false} {
			for typ := 1; typ <= 4; typ++ {
				pts := 1
				for _, s := range shape {
					pts *= logicalSize(cosine, typ, s)
				}
				b := bound32(pts)
				for _, m := range allNorms {
					what := fmt.Sprintf("shape %v cosine %v type %d %v", shape, cosine, typ, m)
					if cosine {
						w.check(t, what+" DCTN32", relErrReal(DCTN32(x, shape, typ, m), DCTN(x64, shape, typ, m)), b)
						w.check(t, what+" IDCTN32", relErrReal(IDCTN32(x, shape, typ, m), IDCTN(x64, shape, typ, m)), b)
					} else {
						w.check(t, what+" DSTN32", relErrReal(DSTN32(x, shape, typ, m), DSTN(x64, shape, typ, m)), b)
						w.check(t, what+" IDSTN32", relErrReal(IDSTN32(x, shape, typ, m), IDSTN(x64, shape, typ, m)), b)
					}
				}
			}
		}
	}
	w.log(t)
}

// TestF32NDNoMutation: no allocating entry point writes into its input.
func TestF32NDNoMutation(t *testing.T) {
	x, _ := signal32(24, 1)
	r, _ := real32(24, 2)
	xs, rs := slices.Clone(x), slices.Clone(r)
	shape := []int{4, 6}
	S := RFFTN32(r, shape)
	Ss := slices.Clone(S)
	FFTN32(x, shape)
	IFFTN32(x, shape)
	FFT32With(x, Options{N: 30})
	IFFT32With(x, Options{})
	RFFT32With(r, Options{N: 10})
	IRFFT32With(S, Options{})
	HFFT32With(S, Options{})
	IHFFT32(r)
	IRFFTN32(S, shape)
	IRFFT2_32(S, [2]int{4, 6})
	DCT32(r, 2, NormOrtho)
	DSTN32(r, shape, 3, NormOrtho)
	if !slices.Equal(x, xs) || !slices.Equal(r, rs) || !slices.Equal(S, Ss) {
		t.Fatal("an entry point modified its input")
	}
}

// TestF32NDEmpty: zero-point transforms return empty, non-nil slices.
func TestF32NDEmpty(t *testing.T) {
	for name, n := range map[string]int{
		"FFT32With":   len(FFT32With(nil, Options{})),
		"RFFT32With":  len(RFFT32With(nil, Options{})),
		"IRFFT32With": len(IRFFT32With([]complex64{1}, Options{})),
		"HFFT32With":  len(HFFT32With(nil, Options{})),
		"HFFT32":      len(HFFT32(nil, 0)),
		"IHFFT32":     len(IHFFT32(nil)),
	} {
		if n != 0 {
			t.Errorf("%s: length %d, want 0", name, n)
		}
	}
	if FFT32With(nil, Options{}) == nil || RFFT32With(nil, Options{}) == nil || IRFFT32With(nil, Options{}) == nil || HFFT32(nil, -1) == nil {
		t.Error("an empty result is nil")
	}
}

// TestF32NDPanics: every invalid argument panics with the package's message.
func TestF32NDPanics(t *testing.T) {
	const want = "fft: "
	huge := []int{1 << 32, 1 << 32}
	c := make([]complex64, 12)
	r := make([]float32, 12)
	cases := map[string]func(){
		"NewPlanN32 overflow":       func() { NewPlanN32(1<<62, 4) },
		"NewPlanN32 zero":           func() { NewPlanN32(3, 0) },
		"FFTN32 overflow":           func() { FFTN32(nil, huge) },
		"FFTN32 mismatch":           func() { FFTN32(c, []int{5, 2}) },
		"IFFTN32 mismatch":          func() { IFFTN32(c, []int{5}) },
		"FFT2_32 overflow":          func() { FFT2_32(nil, [2]int{1 << 32, 1 << 32}) },
		"IFFT2_32 zero":             func() { IFFT2_32(nil, [2]int{0, 3}) },
		"PlanN32.FFT short":         func() { NewPlanN32(3, 4).FFT(c[:11], c) },
		"PlanN32.IFFT long src":     func() { NewPlanN32(3, 4).IFFT(c, make([]complex64, 13)) },
		"PlanN32.FFTNorm bad norm":  func() { NewPlanN32(3, 4).FFTNorm(c, c, Norm(3)) },
		"PlanN32.IFFTNorm bad norm": func() { NewPlanN32(3, 4).IFFTNorm(c, c, Norm(-1)) },
		"NewRealPlan2_32 overflow":  func() { NewRealPlan2_32(1<<62, 4) },
		"RealPlan2_32.RFFT short":   func() { NewRealPlan2_32(3, 4).RFFT(make([]complex64, 9), r[:11]) },
		"RealPlan2_32.RFFT dst":     func() { NewRealPlan2_32(3, 4).RFFT(make([]complex64, 8), r) },
		"RealPlan2_32.IRFFT dst":    func() { NewRealPlan2_32(3, 4).IRFFT(r[:5], make([]complex64, 9)) },
		"RealPlan2_32.IRFFT src":    func() { NewRealPlan2_32(3, 4).IRFFT(r, make([]complex64, 10)) },
		"RealPlan2_32 bad norm":     func() { NewRealPlan2_32(3, 4).IRFFTNorm(r, make([]complex64, 9), Norm(7)) },
		"RFFT2_32 mismatch":         func() { RFFT2_32(r, [2]int{5, 2}) },
		"RFFT2_32 overflow":         func() { RFFT2_32(nil, [2]int{1 << 32, 1 << 32}) },
		"IRFFT2_32 overflow":        func() { IRFFT2_32(nil, [2]int{1 << 32, 1 << 32}) },
		"RFFTN32 empty shape":       func() { RFFTN32([]float32{1}, []int{}) },
		"RFFTN32 mismatch":          func() { RFFTN32(r, []int{5}) },
		"IRFFTN32 empty shape":      func() { IRFFTN32(c, []int{}) },
		"IRFFTN32 overflow":         func() { IRFFTN32(nil, huge) },
		"RFFTN32With no axis":       func() { RFFTN32With(r, []int{12}, Options{Axes: []int{}}) },
		"IRFFTN32With no axis":      func() { IRFFTN32With(c, []int{12}, Options{Axes: []int{}}) },
		"FFTN32With N":              func() { FFTN32With(c, []int{12}, Options{N: 4}) },
		"FFTN32With axis":           func() { FFTN32With(c, []int{12}, Options{Axes: []int{1}}) },
		"IFFTN32With twice":         func() { IFFTN32With(c, []int{3, 4}, Options{Axes: []int{0, -2}}) },
		"FFT2_32With norm":          func() { FFT2_32With(c, [2]int{3, 4}, Options{Norm: 9}) },
		"IFFT2_32With N":            func() { IFFT2_32With(c, [2]int{3, 4}, Options{N: 1}) },
		"RFFT2_32With N":            func() { RFFT2_32With(r, [2]int{3, 4}, Options{N: 1}) },
		"RFFT2_32With axes":         func() { RFFT2_32With(r, [2]int{3, 4}, Options{Axes: []int{5}}) },
		"RFFT2_32With mismatch":     func() { RFFT2_32With(r, [2]int{3, 5}, Options{}) },
		"IRFFT2_32With N":           func() { IRFFT2_32With(c, [2]int{3, 4}, Options{N: 1}) },
		"IRFFT2_32With axes":        func() { IRFFT2_32With(c, [2]int{3, 4}, Options{Axes: []int{0, 0}}) },
		"FFT32With axes":            func() { FFT32With(c, Options{Axes: []int{0}}) },
		"IFFT32With N":              func() { IFFT32With(c, Options{N: -1}) },
		"RFFT32With norm":           func() { RFFT32With(r, Options{Norm: 3}) },
		"IRFFT32With N":             func() { IRFFT32With(c, Options{N: -2}) },
		"HFFT32With axes":           func() { HFFT32With(c, Options{Axes: []int{}}) },
		"IHFFT32With norm":          func() { IHFFT32With(r, Options{Norm: -1}) },
		"DCT32 type":                func() { DCT32(r, 5, NormBackward) },
		"DCT32 length":              func() { DCT32(r[:1], 1, NormBackward) },
		"IDCT32 norm":               func() { IDCT32(r, 2, Norm(4)) },
		"DST32 empty":               func() { DST32(nil, 2, NormBackward) },
		"IDST32 type":               func() { IDST32(r, 0, NormBackward) },
		"NewDCTPlan32":              func() { NewDCTPlan32(1, 1) },
		"NewDSTPlan32":              func() { NewDSTPlan32(4, 9) },
		"DCTPlan32 short":           func() { NewDCTPlan32(12, 2).DCT(r[:3], r, NormBackward) },
		"DCTPlan32 norm":            func() { NewDCTPlan32(12, 2).IDCT(r, r, Norm(5)) },
		"DSTPlan32 short":           func() { NewDSTPlan32(12, 2).IDST(r, r[:4], NormBackward) },
		"DSTPlan32 norm":            func() { NewDSTPlan32(12, 2).DST(r, r, Norm(5)) },
		"DCTN32 mismatch":           func() { DCTN32(r, []int{5}, 2, NormBackward) },
		"IDCTN32 axis below min":    func() { IDCTN32(r, []int{12, 1}, 1, NormBackward) },
		"DSTN32 type":               func() { DSTN32(r, []int{12}, 7, NormBackward) },
		"IDSTN32 norm":              func() { IDSTN32(r, []int{12}, 2, Norm(3)) },
	}
	for name, f := range cases {
		mustPanicWith(t, name, want, f)
	}
}

// TestF32NDLongerSlices: the 1-D plans accept longer slices and leave what
// is past their length alone.
func TestF32NDLongerSlices(t *testing.T) {
	x, x64 := real32(8, 4)
	dst := make([]float32, 10)
	dst[8], dst[9] = 7, 9
	got := NewDCTPlan32(8, 2).DCT(dst, append(x, 5, 5), NormOrtho)
	if len(got) != 8 || dst[8] != 7 || dst[9] != 9 {
		t.Fatalf("DCTPlan32 wrote past its length: %v", dst)
	}
	if e := relErrReal(got, DCT(x64, 2, NormOrtho)); e > bound32(16) {
		t.Fatalf("DCTPlan32 longer slices: error %g", e)
	}
}

// TestF32NDConcurrent runs the plans from several goroutines at once (the
// race detector checks the scratch pools).
func TestF32NDConcurrent(t *testing.T) {
	pn := NewPlanN32(16, 12)
	r2 := NewRealPlan2_32(16, 12)
	dp := NewDCTPlan32(30, 4)
	sp := NewDSTPlan32(31, 4)
	x, x64 := signal32(192, 1)
	r, r64 := real32(192, 2)
	wantN, wantR := FFTN(x64, []int{16, 12}), RFFT2(r64, [2]int{16, 12})
	wantD, wantS := DCT(r64[:30], 4, NormOrtho), DST(r64[:31], 4, NormForward)
	var wg sync.WaitGroup
	errs := make(chan string, 64)
	for g := 0; g < 8; g++ {
		wg.Go(func() {
			for i := 0; i < 20; i++ {
				if e := relErr(pn.FFT(make([]complex64, 192), x), wantN); e > bound32(192) {
					errs <- fmt.Sprint("PlanN32 ", e)
				}
				if e := relErr(r2.RFFT(make([]complex64, r2.SpectrumLen()), r), wantR); e > bound32(192) {
					errs <- fmt.Sprint("RealPlan2_32 ", e)
				}
				if e := relErrReal(dp.DCT(make([]float32, 30), r, NormOrtho), wantD); e > bound32(60) {
					errs <- fmt.Sprint("DCTPlan32 ", e)
				}
				if e := relErrReal(sp.DST(make([]float32, 31), r, NormForward), wantS); e > bound32(62) {
					errs <- fmt.Sprint("DSTPlan32 ", e)
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	var all []string
	for e := range errs {
		all = append(all, e)
	}
	if len(all) > 0 {
		t.Fatal(strings.Join(all, "; "))
	}
}

// TestF32NDSwapNorm pins the Norm arithmetic f32ndSwapNorm relies on.
func TestF32NDSwapNorm(t *testing.T) {
	for _, m := range allNorms {
		s := f32ndSwapNorm(m)
		if s.scale(10, true) != m.scale(10, false) || s.scale(10, false) != m.scale(10, true) {
			t.Errorf("f32ndSwapNorm(%v) = %v", m, s)
		}
	}
	if math.IsNaN(float64(f32ndSqrt2)) || f32ndSqrt2*f32ndSqrt2 < 1.9999 {
		t.Error("f32ndSqrt2")
	}
}

// fuzzF32ND drives the single-precision N-D, Options and DCT/DST entry points
// from FuzzPublicAPI with its fuzzed lengths, shape, Norm, axes and type. Each
// call may panic only with the package's own "fft: ..." message, and each
// valid pair must round-trip within the round-trip form of bound32. a and b
// are already bounded by FuzzPublicAPI.
func fuzzF32ND(t *testing.T, a, b int, seed uint8) {
	norm := Norm(int(seed % 4)) // 3 is not a Norm: it must be refused cleanly
	axesChoices := [][]int{nil, {0}, {-1}, {1, 0}, {}, {2}, {0, -2}}
	axes := axesChoices[int(seed/4)%len(axesChoices)]
	typ := int(seed/3) % 6 // 0 and 5 are not types
	try := func(name string, f func()) {
		t.Helper()
		defer func() {
			if r := recover(); r != nil {
				if msg, _ := r.(string); !strings.HasPrefix(msg, "fft: ") {
					t.Fatalf("%s a=%d b=%d seed=%d: unexpected panic %v", name, a, b, seed, r)
				}
			}
		}()
		f()
	}
	tol := func(n int) float64 { return 4 * bound32(max(n, 2)) }
	if a >= 0 {
		x := make([]complex64, a)
		r := make([]float32, a)
		for i := range x {
			x[i] = complex(float32((i*5+int(seed))%9)-4, float32((i*7+int(seed))%5)-2)
			r[i] = real(x[i])
		}
		r64 := widenReal(r)
		o := Options{N: b, Norm: norm}
		try("FFT32With/IFFT32With", func() {
			X := FFT32With(x, o)
			back := IFFT32With(X, Options{N: len(X), Norm: norm})
			want := make([]complex128, len(back))
			copy(want, widen(x))
			if e := relErr(back, want); e > tol(len(X)) {
				t.Fatalf("FFT32With round trip a=%d N=%d %v: error %g", a, b, norm, e)
			}
		})
		try("RFFT32With/IRFFT32With", func() {
			n := o.N
			if n == 0 {
				n = a
			}
			back := IRFFT32With(RFFT32With(r, o), Options{N: n, Norm: norm})
			want := make([]float64, len(back))
			copy(want, r64)
			if e := relErrReal(back, want); e > tol(n) {
				t.Fatalf("RFFT32With round trip a=%d N=%d %v: error %g", a, b, norm, e)
			}
		})
		try("IHFFT32With/HFFT32With", func() {
			n := o.N
			if n == 0 {
				n = a
			}
			back := HFFT32With(IHFFT32With(r, o), Options{N: n, Norm: norm})
			want := make([]float64, len(back))
			copy(want, r64)
			if e := relErrReal(back, want); e > tol(n) {
				t.Fatalf("IHFFT32With round trip a=%d N=%d %v: error %g", a, b, norm, e)
			}
		})
		try("HFFT32", func() { HFFT32(x, b) })
		try("DCT32/DST32", func() {
			for _, cosine := range []bool{true, false} {
				fwd, inv := DST32, IDST32
				if cosine {
					fwd, inv = DCT32, IDCT32
				}
				if e := relErrReal(inv(fwd(r, typ, norm), typ, norm), r64); e > tol(2*a+2) {
					t.Fatalf("n=%d type %d %v cosine=%v: round trip error %g", a, typ, norm, cosine, e)
				}
			}
		})
	}
	shape := []int{a, b}
	total := 0
	if a > 0 && b > 0 {
		total = a * b
	}
	d := make([]complex64, total)
	rd := make([]float32, total)
	for i := range d {
		d[i] = complex(float32(i%7)-3, float32(i%4))
		rd[i] = float32(i%6) - 2
	}
	o := Options{Axes: axes, Norm: norm}
	try("FFTN32With/IFFTN32With", func() {
		if e := relErr(IFFTN32With(FFTN32With(d, shape, o), shape, o), widen(d)); e > tol(total) {
			t.Fatalf("FFTN32With round trip shape %v axes %v: error %g", shape, axes, e)
		}
	})
	try("FFTN32/IFFTN32", func() {
		if e := relErr(IFFTN32(FFTN32(d, shape), shape), widen(d)); e > tol(total) {
			t.Fatalf("FFTN32 round trip shape %v: error %g", shape, e)
		}
	})
	try("RFFTN32With/IRFFTN32With", func() {
		if e := relErrReal(IRFFTN32With(RFFTN32With(rd, shape, o), shape, o), widenReal(rd)); e > tol(total) {
			t.Fatalf("RFFTN32With round trip shape %v axes %v: error %g", shape, axes, e)
		}
	})
	try("RFFT2_32/IRFFT2_32", func() {
		s2 := [2]int{a, b}
		if e := relErrReal(IRFFT2_32(RFFT2_32(rd, s2), s2), widenReal(rd)); e > tol(total) {
			t.Fatalf("RFFT2_32 round trip shape %v: error %g", shape, e)
		}
	})
	try("DCTN32/IDCTN32", func() {
		if e := relErrReal(IDCTN32(DCTN32(rd, shape, typ, norm), shape, typ, norm), widenReal(rd)); e > tol(4*total) {
			t.Fatalf("DCTN32 round trip shape %v type %d %v: error %g", shape, typ, norm, e)
		}
	})
}
