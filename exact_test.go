package fft

import (
	"strings"
	"testing"
)

// exact returns o with Options.Exact set: the reference arithmetic. (The copy
// of this file that runs against v0.28.16 returns o, which has no such field.)
func exact(o Options) Options { o.Exact = true; return o }

// exactOpts is Options{Exact: true}. The tests that hold a transform to the Go
// reference bit for bit build their plans with it, so that they keep guarding
// that arithmetic when the default stops being it (#121).
var exactOpts = exact(Options{})

// TestExactIsTheDefaultToday: Options.Exact is the reference arithmetic, which
// is what every transform computes by default today, so each transform with
// Exact set gives the default's bits, for every function that takes Options
// (float32 and plan constructors included), on every route (Stockham, the
// iterative power-of-two kernel, Rader, Bluestein, strips, real and 2-D). The
// day the default changes, this test is the one that says so, and the
// comparison moves to the reference (#121 step 3). The golden digests of
// v0.28.16 (exact_golden_test.go) tie both to the released bits.
func TestExactIsTheDefaultToday(t *testing.T) {
	for _, c := range exactCases() {
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
	for _, c := range exactCases() {
		if !sameAny(c.def(), c.exact()) {
			t.Errorf("%s: with FMA available, Options{Exact: true} differs from the default", c.name)
		}
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
