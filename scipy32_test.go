package fft

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// scipyF32 is testdata/scipy_f32.json, written by testdata/gen_scipy_f32.py
// from real scipy: float32/complex64 inputs and the float32/complex64 results
// scipy.fft returns for them. Complex arrays are interleaved.
type scipyF32 struct {
	Scipy string `json:"scipy"`
	Numpy string `json:"numpy"`
	Cases []struct {
		Fn    string    `json:"fn"`
		Norm  string    `json:"norm"`
		Shape []int     `json:"shape"`
		Axes  []int     `json:"axes"`
		N     int       `json:"n"`
		Type  int       `json:"type"`
		X     []float64 `json:"x"`
		Y     []float64 `json:"y"`
	} `json:"cases"`
}

func f32ndReals(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

func f32ndComplexes(v []float64) []complex64 {
	out := make([]complex64, len(v)/2)
	for i := range out {
		out[i] = complex(float32(v[2*i]), float32(v[2*i+1]))
	}
	return out
}

// TestF32NDAgainstScipy compares every single-precision entry point the file
// covers with scipy's own float32 result. Both are within bound32 of the
// exact transform, so they must be within twice that of each other; the
// test also reports how far scipy's float32 result itself is from the
// float64 transform, as a fraction of bound32.
func TestF32NDAgainstScipy(t *testing.T) {
	raw, err := os.ReadFile("testdata/scipy_f32.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref scipyF32
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if len(ref.Cases) != 372 {
		t.Fatalf("reference has %d cases", len(ref.Cases))
	}
	var ours, theirs f32ndWorst
	for _, c := range ref.Cases {
		m := parseNorm(t, c.Norm)
		what := fmt.Sprintf("scipy %s %s %v type %d n %d shape %v axes %v", ref.Scipy, c.Fn, m, c.Type, c.N, c.Shape, c.Axes)
		o := Options{Norm: m, Axes: c.Axes, N: c.N}
		var got, want64 []float64 // interleaved when complex
		var pts int
		switch c.Fn {
		case "fftn", "ifftn":
			x := f32ndComplexes(c.X)
			f32, f64 := FFTN32With, FFTNWith
			if c.Fn == "ifftn" {
				f32, f64 = IFFTN32With, IFFTNWith
			}
			got, want64 = f32ndInterleave(f32(x, c.Shape, o)), f32ndInterleave128(f64(widen(x), c.Shape, o))
			pts = axesPoints(c.Shape, c.Axes)
		case "rfftn":
			x := f32ndReals(c.X)
			got, want64 = f32ndInterleave(RFFTN32With(x, c.Shape, o)), f32ndInterleave128(RFFTNWith(widenReal(x), c.Shape, o))
			pts = axesPoints(c.Shape, c.Axes)
		case "irfftn":
			x := f32ndComplexes(c.X)
			got, want64 = widenReal(IRFFTN32With(x, c.Shape, o)), IRFFTNWith(widen(x), c.Shape, o)
			pts = axesPoints(c.Shape, c.Axes)
		case "fft", "ifft":
			x := f32ndComplexes(c.X)
			f32, f64 := FFT32With, FFTWith
			if c.Fn == "ifft" {
				f32, f64 = IFFT32With, IFFTWith
			}
			got, want64 = f32ndInterleave(f32(x, o)), f32ndInterleave128(f64(widen(x), o))
			pts = len(got) / 2
		case "rfft", "ihfft":
			x := f32ndReals(c.X)
			f32, f64 := RFFT32With, RFFTWith
			if c.Fn == "ihfft" {
				f32, f64 = IHFFT32With, IHFFTWith
			}
			got, want64 = f32ndInterleave(f32(x, o)), f32ndInterleave128(f64(widenReal(x), o))
			pts = len(x)
			if c.N > 0 {
				pts = c.N
			}
		case "irfft", "hfft":
			x := f32ndComplexes(c.X)
			f32, f64 := IRFFT32With, IRFFTWith
			if c.Fn == "hfft" {
				f32, f64 = HFFT32With, HFFTWith
			}
			got, want64 = widenReal(f32(x, o)), f64(widen(x), o)
			pts = len(got)
		case "dct", "idct", "dst", "idst":
			x := f32ndReals(c.X)
			cosine, inverse := c.Fn[1] == 'c' || c.Fn[2] == 'c', c.Fn[0] == 'i'
			got, want64 = widenReal(f32ndR2R(x, cosine, c.Type, m, inverse)), f64R2R(widenReal(x), cosine, c.Type, m, inverse)
			pts = logicalSize(cosine, c.Type, len(x))
		default: // dctn, idctn, dstn, idstn
			x := f32ndReals(c.X)
			fns := map[string]func([]float32, []int, int, Norm) []float32{"dctn": DCTN32, "idctn": IDCTN32, "dstn": DSTN32, "idstn": IDSTN32}
			fns64 := map[string]func([]float64, []int, int, Norm) []float64{"dctn": DCTN, "idctn": IDCTN, "dstn": DSTN, "idstn": IDSTN}
			got, want64 = widenReal(fns[c.Fn](x, c.Shape, c.Type, m)), fns64[c.Fn](widenReal(x), c.Shape, c.Type, m)
			cosine := c.Fn[1] == 'c' || c.Fn[2] == 'c'
			pts = 1
			for _, s := range c.Shape {
				pts *= logicalSize(cosine, c.Type, s)
			}
		}
		if len(got) != len(c.Y) {
			t.Errorf("%s: %d values, scipy has %d", what, len(got), len(c.Y))
			continue
		}
		scipy := widenReal(f32ndReals(c.Y))
		ours.check(t, what, relErrReal(f32ndReals(got), scipy), 2*bound32(max(pts, 2)))
		theirs.check(t, what+" (scipy vs float64)", relErrReal(f32ndReals(c.Y), want64), bound32(max(pts, 2)))
	}
	t.Logf("scipy %s, numpy %s: ours vs scipy float32, worst %.3f of 2·eps32·log2(n)", ref.Scipy, ref.Numpy, ours.frac)
	t.Logf("scipy's own float32 error, worst %.3f of eps32·log2(n) (%s)", theirs.frac, theirs.what)
}

// f32ndInterleave widens x to re0, im0, re1, im1, ...
func f32ndInterleave(x []complex64) []float64 {
	out := make([]float64, 0, 2*len(x))
	for _, v := range x {
		out = append(out, float64(real(v)), float64(imag(v)))
	}
	return out
}

func f32ndInterleave128(x []complex128) []float64 {
	out := make([]float64, 0, 2*len(x))
	for _, v := range x {
		out = append(out, real(v), imag(v))
	}
	return out
}
