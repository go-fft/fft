package fft

import (
	"strconv"
	"strings"
	"testing"
)

// fusedSCase32 reports whether the exactCases case name is a float32 transform of a
// length whose complex64 transform (or, for the real ones, whose half plan) has a
// fused kernel that is on.
func fusedSCase32(name string) bool {
	if !strings.Contains(name, "32") {
		return false
	}
	i := strings.LastIndex(name, "n=")
	if i < 0 {
		return false
	}
	n, err := strconv.Atoi(name[i+2:])
	return err == nil && (fusedSEnabled[n] || fusedSEnabled[n/2])
}

// exact returns o with Options.Exact set: the reference arithmetic. (The copy
// of this file that runs against v0.28.16 returns o, which has no such field.)
func exact(o Options) Options { return exactMode(o) }

// exactMode is what exact applies: Exact, except while a test swaps it for
// NoFMA to run the same case set on the other unfused arithmetic.
var exactMode = func(o Options) Options { o.Exact = true; return o }

// fftRealWithRef is FFTRealWith with the reference arithmetic. (The copy of this
// file that runs against v0.28.16, which has no FFTRealWith, returns FFTWith of
// the promoted signal, which is what FFTReal and the padded forms were.)
func fftRealWithRef(x []float64, o Options) []complex128 { return FFTRealWith(x, exact(o)) }

// exactOpts is Options{Exact: true}. The tests that hold a transform to the Go
// reference bit for bit build their plans with it, so that they keep guarding
// that arithmetic when the default stops being it (#121).
var exactOpts = exact(Options{})

// TestExactIsTheDefaultWhereItDoesNotFuse: Options.Exact is the reference
// arithmetic. Where the default does not fuse (no FMA3, other architectures:
// arith.fused()) it is what every transform computes by default, so each
// transform with Exact set gives the default's bits, for every function that
// takes Options (float32 and plan constructors included), on every route
// (Stockham, the iterative power-of-two kernel, Rader, Bluestein, strips, real
// and 2-D). Where the default fuses (amd64 with FMA3) the two differ in the
// last bits, inside the bound, and TestDefaultFusesWhereItCan says so. The
// golden digests of v0.28.16 (exact_golden_test.go) tie Exact to the released
// bits either way.
func TestExactIsTheDefaultWhereItDoesNotFuse(t *testing.T) {
	if arithDefault.fused() {
		t.Skip("the default fuses on this CPU: see TestDefaultFusesWhereItCan")
	}
	for _, c := range exactCases() {
		if fusedSCase32(c.name) {
			// A float32 transform of a length that has a fused complex64 kernel runs
			// it by default, whatever the arithmetic of the float64 plans: the kernel
			// is not the passes' bits, and Exact is (TestFusedSExactAndInternalPlans
			// KeepTheirBits).
			continue
		}
		if digestCase(c.name, c.def()) != digestCase(c.name, c.exact()) { // two cases up to a zero's sign (zeroSignCase)
			t.Errorf("%s: Options{Exact: true} differs from the default", c.name)
		}
	}
}

// TestDefaultFusesWhereItCan: the default arithmetic is the fused one where the
// CPU has FMA3, and differs from Exact in the last bits somewhere over the case
// set; NoFMA and Exact never fuse; FMA is the default there.
func TestDefaultFusesWhereItCan(t *testing.T) {
	isolatePlans(t)
	// As shipped: the default fuses exactly where FMA3 runs.
	if !defaultFuses || arithDefault.fused() != fmaAvailable {
		t.Fatalf("defaultFuses = %v, default fuses = %v, FMA3 available = %v", defaultFuses, arithDefault.fused(), fmaAvailable)
	}
	// With FMA forced available (any architecture) and the default's fusing on.
	forceFMA(t)
	defaultFuses = true
	forgetAllPlans()
	if !cachedPlanA(1000, arithDefault).sk.fma {
		t.Fatal("the default plan of 1000 points does not fuse")
	}
	if cachedPlanA(1000, arithPlain).sk.fma || cachedPlanA(1000, arithExact).sk.fma {
		t.Fatal("a NoFMA or Exact plan fuses")
	}
	if (Options{}).arith() != arithDefault || (Options{NoFMA: true}).arith() != arithPlain || (Options{FMA: true}).arith() != arithFMA {
		t.Fatal("Options.arith")
	}
	differ := 0
	for _, c := range exactCases() {
		if !strings.Contains(c.name, "32") && !strings.HasPrefix(c.name, "FFTRealWith") && !sameAny(c.def(), c.exact()) {
			differ++
		}
	}
	if differ == 0 {
		t.Error("the default fuses and equals Exact on every case: the flip did nothing")
	}
}

// TestExactIgnoresFMAAvailability: Exact never builds an FMA plan, whatever
// the CPU offers, and its bits do not depend on whether FMA is available:
// every case gives the same digest with FMA switched off and forced on.
func TestExactIgnoresFMAAvailability(t *testing.T) {
	isolatePlans(t)
	cases := exactCases()
	defer func(a bool) { fmaAvailable = a }(fmaAvailable)
	fmaAvailable = false
	forgetAllPlans()
	off := make([]string, len(cases))
	for i, c := range cases {
		off[i] = digestAny(c.exact())
	}
	forceFMA(t)
	for i, c := range cases {
		if got := digestAny(c.exact()); got != off[i] {
			t.Errorf("%s: Exact's bits depend on FMA being available", c.name)
		}
	}
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

// TestNoFMAAndFMARefused: NoFMA and FMA together panic, naming the function, from
// the validators every entry point shares (1-D, N-D, plan constructors and the
// float32 ones); NoFMA with Exact is redundant and accepted.
func TestNoFMAAndFMARefused(t *testing.T) {
	both := Options{NoFMA: true, FMA: true}
	x := make([]complex128, 8)
	d := make([]complex128, 16)
	x32 := make([]complex64, 8)
	for name, f := range map[string]func(){
		"FFTWith":         func() { FFTWith(x, both) },
		"FFTNWith":        func() { FFTNWith(d, []int{4, 4}, both) },
		"NewPlanWith":     func() { NewPlanWith(8, both) },
		"NewRealPlanWith": func() { NewRealPlanWith(8, both) },
		"FFT32With":       func() { FFT32With(x32, both) },
	} {
		func() {
			defer func() {
				if r, _ := recover().(string); !strings.Contains(r, name+": Options.NoFMA and Options.FMA are mutually exclusive") {
					t.Errorf("%s: panic %q", name, r)
				}
			}()
			f()
		}()
	}
	if got := FFTWith(x, Options{NoFMA: true, Exact: true}); len(got) != 8 {
		t.Fatal("NoFMA with Exact is accepted")
	}
}
