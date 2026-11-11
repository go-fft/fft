package fft

import (
	"math"
	"strings"
	"testing"
)

// exactOpts is Options{Exact: true}: the reference arithmetic. The tests that
// hold a transform to the Go reference bit for bit build their plans with it,
// so that they keep guarding that arithmetic when the default stops being it
// (#121).
var exactOpts = Options{Exact: true}

func sameReal(a, b []float64) bool {
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return false
		}
	}
	return len(a) == len(b)
}

func sameReal32(a, b []float32) bool {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return len(a) == len(b)
}

// exactCase is one transform called with and without Options.Exact.
type exactCase struct {
	name       string
	def, exact func() any
}

func exactCases(t *testing.T) []exactCase {
	t.Helper()
	var cs []exactCase
	add := func(name string, def, exact func() any) { cs = append(cs, exactCase{name, def, exact}) }
	for _, n := range []int{1, 2, 17, 64, 360, 1000, 1009, 2018, 4096, 4099} {
		x := fmaSignal(n, uint64(n))
		r := fmaReal(n, uint64(n))
		h := RFFT(r)
		x32 := make([]complex64, n)
		r32 := make([]float32, n)
		for i := range x {
			x32[i] = complex64(x[i])
			r32[i] = float32(r[i])
		}
		h32 := RFFT32(r32)
		add("FFT", func() any { return FFT(x) }, func() any { return FFTWith(x, exactOpts) })
		add("IFFT", func() any { return IFFT(x) }, func() any { return IFFTWith(x, exactOpts) })
		add("RFFT", func() any { return RFFT(r) }, func() any { return RFFTWith(r, exactOpts) })
		add("IRFFT", func() any { return IRFFT(h, n) }, func() any { return IRFFTWith(h, Options{N: n, Exact: true}) })
		add("HFFT", func() any { return HFFT(h, n) }, func() any { return HFFTWith(h, Options{N: n, Exact: true}) })
		add("IHFFT", func() any { return IHFFT(r) }, func() any { return IHFFTWith(r, exactOpts) })
		add("NewPlanWith", func() any { return NewPlan(n).FFT(make([]complex128, n), x) },
			func() any { return NewPlanWith(n, exactOpts).FFT(make([]complex128, n), x) })
		add("NewRealPlanWith", func() any { return NewRealPlan(n).RFFT(make([]complex128, n/2+1), r) },
			func() any { return NewRealPlanWith(n, exactOpts).RFFT(make([]complex128, n/2+1), r) })
		add("FFT32With", func() any { return FFT32(x32) }, func() any { return FFT32With(x32, exactOpts) })
		add("IFFT32With", func() any { return IFFT32(x32) }, func() any { return IFFT32With(x32, exactOpts) })
		add("RFFT32With", func() any { return RFFT32(r32) }, func() any { return RFFT32With(r32, exactOpts) })
		add("IRFFT32With", func() any { return IRFFT32(h32, n) }, func() any { return IRFFT32With(h32, Options{N: n, Exact: true}) })
		add("HFFT32With", func() any { return HFFT32(h32, n) }, func() any { return HFFT32With(h32, Options{N: n, Exact: true}) })
		add("IHFFT32With", func() any { return IHFFT32(r32) }, func() any { return IHFFT32With(r32, exactOpts) })
	}
	for _, sh := range [][]int{{12, 20}, {8, 8, 8}, {17, 9}, {30, 31}} {
		size := shapeProduct(sh...)
		x := fmaSignal(size, uint64(size))
		r := fmaReal(size, uint64(size))
		sp := RFFTN(r, sh)
		x32 := make([]complex64, size)
		r32 := make([]float32, size)
		for i := range x {
			x32[i] = complex64(x[i])
			r32[i] = float32(r[i])
		}
		sp32 := RFFTN32(r32, sh)
		add("FFTN", func() any { return FFTN(x, sh) }, func() any { return FFTNWith(x, sh, exactOpts) })
		add("IFFTN", func() any { return IFFTN(x, sh) }, func() any { return IFFTNWith(x, sh, exactOpts) })
		add("RFFTN", func() any { return RFFTN(r, sh) }, func() any { return RFFTNWith(r, sh, exactOpts) })
		add("IRFFTN", func() any { return IRFFTN(sp, sh) }, func() any { return IRFFTNWith(sp, sh, exactOpts) })
		add("NewPlanNWith", func() any { return NewPlanN(sh...).FFT(make([]complex128, size), x) },
			func() any { return NewPlanNWith(sh, exactOpts).FFT(make([]complex128, size), x) })
		add("FFTN32With", func() any { return FFTN32(x32, sh) }, func() any { return FFTN32With(x32, sh, exactOpts) })
		add("IFFTN32With", func() any { return IFFTN32(x32, sh) }, func() any { return IFFTN32With(x32, sh, exactOpts) })
		add("RFFTN32With", func() any { return RFFTN32(r32, sh) }, func() any { return RFFTN32With(r32, sh, exactOpts) })
		add("IRFFTN32With", func() any { return IRFFTN32(sp32, sh) }, func() any { return IRFFTN32With(sp32, sh, exactOpts) })
		if len(sh) == 2 {
			s2 := [2]int{sh[0], sh[1]}
			add("FFT2", func() any { return FFT2(x, s2) }, func() any { return FFT2With(x, s2, exactOpts) })
			add("IFFT2", func() any { return IFFT2(x, s2) }, func() any { return IFFT2With(x, s2, exactOpts) })
			add("RFFT2", func() any { return RFFT2(r, s2) }, func() any { return RFFT2With(r, s2, exactOpts) })
			add("IRFFT2", func() any { return IRFFT2(sp, s2) }, func() any { return IRFFT2With(sp, s2, exactOpts) })
			add("NewRealPlan2With", func() any { return NewRealPlan2(s2[0], s2[1]).RFFT(make([]complex128, s2[0]*(s2[1]/2+1)), r) },
				func() any {
					return NewRealPlan2With(s2[0], s2[1], exactOpts).RFFT(make([]complex128, s2[0]*(s2[1]/2+1)), r)
				})
			add("FFT2_32With", func() any { return FFT2_32(x32, s2) }, func() any { return FFT2_32With(x32, s2, exactOpts) })
			add("IFFT2_32With", func() any { return IFFT2_32(x32, s2) }, func() any { return IFFT2_32With(x32, s2, exactOpts) })
			add("RFFT2_32With", func() any { return RFFT2_32(r32, s2) }, func() any { return RFFT2_32With(r32, s2, exactOpts) })
			add("IRFFT2_32With", func() any { return IRFFT2_32(sp32, s2) }, func() any { return IRFFT2_32With(sp32, s2, exactOpts) })
		}
	}
	return cs
}

func sameAny(a, b any) bool {
	switch x := a.(type) {
	case []complex128:
		return sameAll(x, b.([]complex128))
	case []float64:
		return sameReal(x, b.([]float64))
	case []complex64:
		y := b.([]complex64)
		for i := range x {
			if math.Float32bits(real(x[i])) != math.Float32bits(real(y[i])) || math.Float32bits(imag(x[i])) != math.Float32bits(imag(y[i])) {
				return false
			}
		}
		return len(x) == len(y)
	case []float32:
		return sameReal32(x, b.([]float32))
	}
	panic("sameAny: unknown type")
}

// TestExactIsTheDefaultToday: Options.Exact is the reference arithmetic, which
// is what every transform computes by default today, so each transform with
// Exact set gives the default's bits, for every function that takes Options
// (float32 and plan constructors included), on every route (Stockham, the
// iterative power-of-two kernel, Rader, Bluestein, strips, real and 2-D). The
// day the default changes, this test is the one that says so, and the
// comparison moves to the reference (#121 step 3).
func TestExactIsTheDefaultToday(t *testing.T) {
	for _, c := range exactCases(t) {
		if !sameAny(c.def(), c.exact()) {
			t.Errorf("%s: Options{Exact: true} differs from the default", c.name)
		}
	}
}

// TestExactIgnoresFMAAvailability: Exact never builds an FMA plan, whatever
// the CPU offers, and with FMA forced available it still gives the bits of the
// default plan, which an FMA request does not.
func TestExactIgnoresFMAAvailability(t *testing.T) {
	forceFMA(t)
	for _, n := range []int{64, 256, 1000, 1009, 4099} {
		p := NewPlanWith(n, exactOpts)
		if p.sk != nil && p.sk.fma || p.rader != nil && p.rader.fma || p.bluestein != nil && p.bluestein.fma {
			t.Fatalf("n=%d: an Exact plan fuses", n)
		}
		if r := NewRealPlanWith(n, exactOpts); r.fma {
			t.Fatalf("n=%d: an Exact real plan is an FMA plan", n)
		}
		if exactOpts.fma() || !(Options{FMA: true}).fma() {
			t.Fatal("fma(): Exact must be off and FMA on while FMA is available")
		}
	}
	if rp2 := NewRealPlan2With(8, 8, exactOpts); NewPlanNWith([]int{8, 8}, exactOpts).fma || rp2.col.fma || rp2.row.fma {
		t.Fatal("an Exact N-D plan is an FMA plan")
	}
	for _, c := range exactCases(t) {
		if !sameAny(c.def(), c.exact()) {
			t.Errorf("%s: with FMA available, Options{Exact: true} differs from the default", c.name)
		}
	}
}

// TestExactKeepsOtherOptions: Exact is orthogonal to N, Norm and Axes.
func TestExactKeepsOtherOptions(t *testing.T) {
	x := fmaSignal(100, 3)
	for _, norm := range []Norm{NormBackward, NormOrtho, NormForward} {
		a := FFTWith(x, Options{N: 128, Norm: norm})
		b := FFTWith(x, Options{N: 128, Norm: norm, Exact: true})
		if !sameAll(a, b) {
			t.Errorf("FFTWith N=128 %v: Exact changed the result", norm)
		}
	}
	d := fmaSignal(12*20, 4)
	if !sameAll(FFTNWith(d, []int{12, 20}, Options{Axes: []int{0}}), FFTNWith(d, []int{12, 20}, Options{Axes: []int{0}, Exact: true})) {
		t.Error("FFTNWith Axes: Exact changed the result")
	}
}

// TestExactAndFMARefused: asking for both arithmetics panics, naming the
// function, before anything is allocated, from every entry point that takes
// Options.
func TestExactAndFMARefused(t *testing.T) {
	both := Options{Exact: true, FMA: true}
	x := make([]complex128, 8)
	r := make([]float64, 8)
	x32 := make([]complex64, 8)
	r32 := make([]float32, 8)
	sh, s2 := []int{4, 4}, [2]int{4, 4}
	d := make([]complex128, 16)
	rd := make([]float64, 16)
	d32 := make([]complex64, 16)
	rd32 := make([]float32, 16)
	for name, f := range map[string]func(){
		"FFTWith":          func() { FFTWith(x, both) },
		"IFFTWith":         func() { IFFTWith(x, both) },
		"RFFTWith":         func() { RFFTWith(r, both) },
		"IRFFTWith":        func() { IRFFTWith(x, both) },
		"HFFTWith":         func() { HFFTWith(x, both) },
		"IHFFTWith":        func() { IHFFTWith(r, both) },
		"FFTNWith":         func() { FFTNWith(d, sh, both) },
		"IFFTNWith":        func() { IFFTNWith(d, sh, both) },
		"FFT2With":         func() { FFT2With(d, s2, both) },
		"IFFT2With":        func() { IFFT2With(d, s2, both) },
		"RFFTNWith":        func() { RFFTNWith(rd, sh, both) },
		"IRFFTNWith":       func() { IRFFTNWith(d, sh, both) },
		"RFFT2With":        func() { RFFT2With(rd, s2, both) },
		"IRFFT2With":       func() { IRFFT2With(d, s2, both) },
		"NewPlanWith":      func() { NewPlanWith(8, both) },
		"NewRealPlanWith":  func() { NewRealPlanWith(8, both) },
		"NewPlanNWith":     func() { NewPlanNWith(sh, both) },
		"NewRealPlan2With": func() { NewRealPlan2With(4, 4, both) },
		"FFT32With":        func() { FFT32With(x32, both) },
		"IFFT32With":       func() { IFFT32With(x32, both) },
		"RFFT32With":       func() { RFFT32With(r32, both) },
		"IRFFT32With":      func() { IRFFT32With(x32, both) },
		"HFFT32With":       func() { HFFT32With(x32, both) },
		"IHFFT32With":      func() { IHFFT32With(r32, both) },
		"FFTN32With":       func() { FFTN32With(d32, sh, both) },
		"IFFTN32With":      func() { IFFTN32With(d32, sh, both) },
		"FFT2_32With":      func() { FFT2_32With(d32, s2, both) },
		"IFFT2_32With":     func() { IFFT2_32With(d32, s2, both) },
		"RFFTN32With":      func() { RFFTN32With(rd32, sh, both) },
		"IRFFTN32With":     func() { IRFFTN32With(d32, sh, both) },
		"RFFT2_32With":     func() { RFFT2_32With(rd32, s2, both) },
		"IRFFT2_32With":    func() { IRFFT2_32With(d32, s2, both) },
	} {
		func() {
			defer func() {
				if r, _ := recover().(string); !strings.Contains(r, name+": Options.Exact and Options.FMA are mutually exclusive") {
					t.Errorf("%s: panic %q", name, r)
				}
			}()
			f()
		}()
	}
}
