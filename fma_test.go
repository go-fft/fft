package fft

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"strings"
	"testing"
)

// fmaSweep is the lengths lo .. hi a sweep of FMA plans runs: one smooth
// length per route (routeSample: its strips' and its lines' factorizations,
// and the passes its FMA plan fuses), and of the others (Rader and
// Bluestein) every one to 150 and one in 13 beyond. Every length took 30 s
// of the Windows suite and 119 s of the macOS one (race and coverage on).
func fmaSweep(lo, hi int) []int {
	var all []int
	for n := lo; n <= hi; n++ {
		all = append(all, n)
	}
	return routeSample(all, func(n int) string {
		if !factorsAreSmall(n) {
			if n <= 150 || n%13 == 0 {
				return fmt.Sprintf("n=%d", n)
			}
			return "dropped"
		}
		f := fmaFactors(n)
		return fmt.Sprintf("%s|%s|%s|%b", lengthClass(n), passKey(n, skFactorizeOrder(n, compOddFirst)), passKey(n, f), fmaFuse(n, f))
	})
}

// The opt-in FMA mode (Round 41) on every architecture: the option's
// routing, its caches, and that every path honours it. forceFMA (per
// architecture) builds FMA plans whatever the CPU.

// forgetFMAPlans empties the FMA plans' caches.
func forgetFMAPlans() {
	planMu.Lock()
	planCacheFMA, planNo16CacheFMA, clRowCacheFMA = map[int]*Plan{}, map[int]*Plan{}, map[int]*Plan{}
	planMu.Unlock()
	realPlanMu.Lock()
	realPlanCacheFMA = map[int]*RealPlan{}
	realPlanMu.Unlock()
	realPlan2Mu.Lock()
	realPlan2CacheFMA = map[[2]int]*RealPlan2{}
	realPlan2Mu.Unlock()
	planNMu.Lock()
	planNCacheFMA = map[string]*PlanN{}
	planNMu.Unlock()
	planNAxesMu.Lock()
	planNAxesCache = map[string]*PlanN{}
	planNAxesMu.Unlock()
}

func fmaSignal(n int, seed uint64) []complex128 {
	r := rand.New(rand.NewPCG(seed, 41))
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(2*r.Float64()-1, 2*r.Float64()-1)
	}
	return x
}

func fmaReal(n int, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 42))
	x := make([]float64, n)
	for i := range x {
		x[i] = 2*r.Float64() - 1
	}
	return x
}

// relRMS is ‖got − want‖ / ‖want‖.
func relRMS(got, want []complex128) float64 {
	var e, s float64
	for i := range want {
		d := got[i] - want[i]
		e += real(d)*real(d) + imag(d)*imag(d)
		s += real(want[i])*real(want[i]) + imag(want[i])*imag(want[i])
	}
	return math.Sqrt(e / s)
}

func sameAll(a, b []complex128) bool {
	for i := range a {
		if !sameBits(a[i], b[i]) {
			return false
		}
	}
	return len(a) == len(b)
}

func TestFMAOn(t *testing.T) {
	for _, c := range []struct{ fma, avail, want bool }{{false, false, false}, {false, true, false}, {true, false, false}, {true, true, true}} {
		if got := fmaOn(c.fma, c.avail); got != c.want {
			t.Errorf("fmaOn(%v, %v) = %v", c.fma, c.avail, got)
		}
	}
	if fmaRadix(7) || fmaRadix(11) || fmaRadix(13) || !fmaRadix(16) || !fmaRadix(20) {
		t.Error("fmaRadix: the radices with amd64 kernels, and no other")
	}
}

// TestFMAPlanOptionsRefused: only FMA applies to a plan constructor.
func TestFMAPlanOptionsRefused(t *testing.T) {
	for _, o := range []Options{{N: 4}, {Norm: NormOrtho}, {Axes: []int{0}}} {
		for name, f := range map[string]func(){
			"NewPlanWith":      func() { NewPlanWith(8, o) },
			"NewRealPlanWith":  func() { NewRealPlanWith(8, o) },
			"NewPlanNWith":     func() { NewPlanNWith([]int{4, 4}, o) },
			"NewRealPlan2With": func() { NewRealPlan2With(4, 4, o) },
		} {
			func() {
				defer func() {
					if r, _ := recover().(string); !strings.Contains(r, name+": only Options.FMA applies to a plan") {
						t.Errorf("%s(%+v): panic %q", name, o, r)
					}
				}()
				f()
			}()
		}
	}
}

// TestFMAUnavailableIsDefault: where the CPU has no FMA, an FMA request
// builds the default plan, and every transform is the default one bit for
// bit. Forced both ways, so it does not depend on the machine.
func TestFMAUnavailableIsDefault(t *testing.T) {
	defer func(a bool) { fmaAvailable = a; forgetFMAPlans() }(fmaAvailable)
	fmaAvailable = false
	forgetFMAPlans()
	o := Options{FMA: true}
	for _, n := range []int{64, 1000, 1009, 10007, 4096} {
		x := fmaSignal(n, uint64(n))
		if p := NewPlanWith(n, o); p.sk != nil && p.sk.fma || p.rader != nil && p.rader.fma || p.bluestein != nil && p.bluestein.fma {
			t.Fatalf("n=%d: an FMA plan was built without FMA", n)
		}
		if !sameAll(FFTWith(x, o), FFT(x)) || !sameAll(IFFTWith(x, o), IFFT(x)) {
			t.Fatalf("n=%d: FFTWith/IFFTWith with FMA unavailable differ from the default", n)
		}
		r := fmaReal(n, 1)
		if !sameAll(RFFTWith(r, o), RFFT(r)) || NewRealPlanWith(n, o).fma {
			t.Fatalf("n=%d: RFFTWith with FMA unavailable differs", n)
		}
	}
	if NewPlanNWith([]int{8, 8}, o).fma || cachedPlanNF([]int{8, 8}, o.fma()) != cachedPlanN([]int{8, 8}) {
		t.Fatal("NewPlanNWith with FMA unavailable is not the default plan")
	}
	if len(planCacheFMA) != 0 || len(realPlanCacheFMA) != 0 || len(planNCacheFMA) != 0 {
		t.Fatal("an FMA cache was filled with FMA unavailable")
	}
}

// TestFMAPlansCachedApart: an FMA plan and the default one never share a
// cache entry, at any level, and the FMA one fuses.
func TestFMAPlansCachedApart(t *testing.T) {
	forceFMA(t)
	for _, n := range []int{256, 1000, 1009, 10007} {
		d, f := cachedPlanF(n, false), cachedPlanF(n, true)
		if d == f || cachedPlanF(n, true) != f || cachedPlan(n) != d {
			t.Fatalf("n=%d: the FMA and default plans share an entry, or a cache returns another", n)
		}
		switch {
		case f.sk != nil && (!f.sk.fma || d.sk.fma):
			t.Fatalf("n=%d: Stockham plan fma flags %v %v", n, d.sk.fma, f.sk.fma)
		case f.rader != nil && (!f.rader.fma || d.rader.fma):
			t.Fatalf("n=%d: Rader fma flags", n)
		case f.bluestein != nil && (!f.bluestein.fma || d.bluestein.fma):
			t.Fatalf("n=%d: Bluestein fma flags", n)
		}
	}
	if cachedRealPlanF(1000, true) == cachedRealPlan(1000) || !cachedRealPlanF(1000, true).half.sk.fma || cachedRealPlan(1000).half.sk.fma {
		t.Fatal("real plans: FMA and default mixed")
	}
	if cachedRealPlan2F(8, 16, true) == cachedRealPlan2(8, 16) || !cachedRealPlan2F(8, 16, true).col.fma {
		t.Fatal("real 2-D plans: FMA and default mixed")
	}
	if cachedPlanNF([]int{16, 16}, true) == cachedPlanN([]int{16, 16}) || planNFor([]int{16, 16}, []int{0}, true) == planNFor([]int{16, 16}, []int{0}, false) {
		t.Fatal("N-D plans: FMA and default mixed")
	}
	if p := cachedPlanNo16F(48, true); p == cachedPlanNo16(48) || !p.sk.fma || cachedPlanNo16F(48, true) != p {
		t.Fatal("no-16 plans: FMA and default mixed")
	}
}

// fmaLengths are lengths for every engine: powers of two, composites with
// every radix, radix-7/11/13 passes, Rader primes, Bluestein lengths (a
// prime factor no pass handles: 17·4, 9·19, 2·10007).
var fmaLengths = []int{2, 3, 8, 16, 32, 60, 64, 68, 96, 100, 120, 128, 144, 160, 171, 180, 192, 240, 256, 288, 320, 343, 360,
	400, 480, 500, 512, 600, 720, 768, 960, 1000, 1001, 1009, 1024, 1080, 1296, 1920, 2048, 2187, 3125, 4096, 6000, 8192, 10007, 20014}

// TestFMAMatchesNaive: every FMA plan transforms correctly, forward and
// inverse, its products fused (its bits differ from the default plan's on a
// random input wherever a pass multiplies), and its error is the default's.
func TestFMAMatchesNaive(t *testing.T) {
	forceFMA(t)
	differ := 0
	for _, n := range fmaLengths {
		x := fmaSignal(n, uint64(n))
		f, d := newPlanF(n, true), newPlanF(n, false)
		got := f.FFT(make([]complex128, n), x)
		def := d.FFT(make([]complex128, n), x)
		want := def // beyond 2,048 points the default plan is the reference
		if n <= 2048 {
			want = fmaNaive(x)
		}
		ef, ed := relRMS(got, want), relRMS(def, want)
		if ef > 1e-14 || n <= 2048 && ef > 2*ed+1e-16 {
			t.Fatalf("n=%d: FMA error %.3g, default %.3g", n, ef, ed)
		}
		if !sameAll(got, def) {
			differ++
		}
		back := f.IFFT(make([]complex128, n), got)
		if e := relRMS(back, x); e > 1e-13 {
			t.Fatalf("n=%d: FMA IFFT(FFT(x)) error %.3g", n, e)
		}
	}
	// arm64 keeps its NEON split passes (fmaSplit false) and its Go passes
	// are fused by the compiler already: the count means something on amd64.
	if fmaSplit && differ < len(fmaLengths)/2 {
		t.Fatalf("only %d of %d FMA plans differ from the default: the products are not fused", differ, len(fmaLengths))
	}
}

// TestFMAEveryEntryPoint: each With function and plan constructor honours
// FMA: its result is the FMA plans' composition, bit for bit, and it is
// accurate.
func TestFMAEveryEntryPoint(t *testing.T) {
	forceFMA(t)
	o := Options{FMA: true}
	for _, n := range []int{240, 1000, 1009, 1024, 10007, 68} {
		x := fmaSignal(n, 7)
		p := newPlanF(n, true)
		if !sameAll(FFTWith(x, o), p.FFT(make([]complex128, n), x)) || !sameAll(NewPlanWith(n, o).FFT(make([]complex128, n), x), p.FFT(make([]complex128, n), x)) {
			t.Fatalf("n=%d: FFTWith or NewPlanWith does not run the FMA plan", n)
		}
		if !sameAll(IFFTWith(x, o), p.IFFT(make([]complex128, n), x)) {
			t.Fatalf("n=%d: IFFTWith does not run the FMA plan", n)
		}
		r := fmaReal(n, 3)
		rp := newRealPlanF(n, true)
		spec := rp.RFFT(make([]complex128, n/2+1), r)
		if !sameAll(RFFTWith(r, o), spec) || !sameAll(NewRealPlanWith(n, o).RFFT(make([]complex128, n/2+1), r), spec) {
			t.Fatalf("n=%d: RFFTWith or NewRealPlanWith does not run the FMA plan", n)
		}
		if e := relRMS(spec, RFFT(r)); e > 1e-13 {
			t.Fatalf("n=%d: FMA RFFT error %.3g against the default", n, e)
		}
		inv := rp.IRFFT(make([]float64, n), spec)
		if g := IRFFTWith(spec, Options{N: n, FMA: true}); !sameFloats(g, inv) {
			t.Fatalf("n=%d: IRFFTWith does not run the FMA plan", n)
		}
		h := HFFTWith(spec, Options{N: n, FMA: true})
		want := irfftF(conjAll(spec), n, true)
		for i := range want {
			want[i] *= float64(n)
		}
		if !sameFloats(h, want) {
			t.Fatalf("n=%d: HFFTWith does not run the FMA plan", n)
		}
		ih := IHFFTWith(r, o)
		sc := append([]complex128(nil), spec...)
		scaleComplex(sc, 1/float64(n))
		for k := range ih {
			if !sameBits(ih[k], cmplx.Conj(sc[k])) {
				t.Fatalf("n=%d: IHFFTWith does not run the FMA plan", n)
			}
		}
	}
	for _, sh := range [][]int{{16, 48}, {171, 4}, {68, 9}, {32, 1, 20}} {
		size := shapeProduct(sh...)
		x := fmaSignal(size, 9)
		p := newPlanNF(sh, true)
		want := p.FFT(make([]complex128, size), x)
		if !sameAll(FFTNWith(x, sh, o), want) || !sameAll(NewPlanNWith(sh, o).FFT(make([]complex128, size), x), want) {
			t.Fatalf("%v: FFTNWith or NewPlanNWith does not run the FMA plan", sh)
		}
		if e := relRMS(want, FFTN(x, sh)); e > 1e-13 {
			t.Fatalf("%v: FMA FFTN error %.3g against the default", sh, e)
		}
		if !sameAll(IFFTNWith(x, sh, o), p.IFFT(make([]complex128, size), x)) {
			t.Fatalf("%v: IFFTNWith does not run the FMA plan", sh)
		}
		for ax, pl := range p.axes {
			if pl != nil && pl.sk != nil && !pl.sk.fma || pl != nil && pl.rader != nil && !pl.rader.fma || pl != nil && pl.bluestein != nil && !pl.bluestein.fma {
				t.Fatalf("%v: axis %d is not an FMA plan", sh, ax)
			}
		}
	}
	x := fmaSignal(12*20, 11)
	if !sameAll(FFT2With(x, [2]int{12, 20}, o), newPlanNF([]int{12, 20}, true).FFT(make([]complex128, 240), x)) ||
		!sameAll(IFFT2With(x, [2]int{12, 20}, o), newPlanNF([]int{12, 20}, true).IFFT(make([]complex128, 240), x)) {
		t.Fatal("FFT2With or IFFT2With does not run the FMA plan")
	}
	sub := FFTNWith(x, []int{12, 20}, Options{Axes: []int{0}, FMA: true})
	if !sameAll(sub, FFTNWith(x, []int{12, 20}, Options{Axes: []int{0}, FMA: true})) || relRMS(sub, FFTNWith(x, []int{12, 20}, Options{Axes: []int{0}})) > 1e-13 {
		t.Fatal("FFTNWith over a subset of axes with FMA")
	}
	r := fmaReal(12*20, 5)
	rp2 := newRealPlan2F(12, 20, true)
	spec := rp2.RFFT(make([]complex128, rp2.SpectrumLen()), r)
	if !sameAll(RFFT2With(r, [2]int{12, 20}, o), spec) || !sameAll(NewRealPlan2With(12, 20, o).RFFT(make([]complex128, rp2.SpectrumLen()), r), spec) {
		t.Fatal("RFFT2With or NewRealPlan2With does not run the FMA plan")
	}
	if !sameFloats(IRFFT2With(spec, [2]int{12, 20}, o), rp2.IRFFT(make([]float64, 240), spec)) {
		t.Fatal("IRFFT2With does not run the FMA plan")
	}
	rn := RFFTNWith(r, []int{12, 20}, o)
	if relRMS(rn, RFFTN(r, []int{12, 20})) > 1e-13 || !sameAll(rn, RFFTNWith(r, []int{12, 20}, o)) {
		t.Fatal("RFFTNWith with FMA")
	}
	back := IRFFTNWith(rn, []int{12, 20}, o)
	for i := range r {
		if math.Abs(back[i]-r[i]) > 1e-13 {
			t.Fatal("IRFFTNWith(RFFTNWith(x)) with FMA")
		}
	}
	if rs := RFFT2With(r, [2]int{12, 20}, Options{Axes: []int{0, 1}, FMA: true}); !sameAll(rs, rn) {
		t.Fatal("RFFT2With with axes and FMA")
	}
	if b := IRFFT2With(rn, [2]int{12, 20}, Options{Axes: []int{0, 1}, FMA: true}); !sameFloats(b, back) {
		t.Fatal("IRFFT2With with axes and FMA")
	}
}

func sameFloats(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return false
		}
	}
	return true
}

// TestFMAStripsMatchLines: the strip axes of an FMA plan, batched passes,
// are bit-identical to its lines gathered and transformed one by one, for
// every length up to 600 and the Bluestein and radix-9·p lengths; and every
// length up to 2000 is correct as a strip axis (TestNDStripAxisEveryLength
// with FMA).
func TestFMAStripsMatchLines(t *testing.T) {
	forceFMA(t)
	defer func(s bool) { stripAxes = s }(stripAxes)
	lens := append([]int{153, 171, 207, 1000, 1024, 1080}, fmaSweep(2, 600)...)
	for _, n := range lens {
		sh := []int{n, 3}
		x := fmaSignal(3*n, uint64(n))
		stripAxes = true
		a := newPlanNF(sh, true)
		got := a.FFT(make([]complex128, 3*n), x)
		inv := a.IFFT(make([]complex128, 3*n), x)
		stripAxes = false
		b := newPlanNF(sh, true)
		if !sameAll(got, b.FFT(make([]complex128, 3*n), x)) || !sameAll(inv, b.IFFT(make([]complex128, 3*n), x)) {
			t.Fatalf("n=%d: FMA strips differ from FMA lines", n)
		}
		if a.axes[0].sk != nil && !a.axes[0].sk.fma {
			t.Fatalf("n=%d: the strip axis is not an FMA plan", n)
		}
	}
	stripAxes = true
	for _, n := range fmaSweep(2, 2000) {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(float64(i%5), float64(i%3))
		}
		want := cachedPlanF(n, true).FFT(make([]complex128, n), x)
		d := make([]complex128, 2*n)
		for i := range x {
			d[2*i], d[2*i+1] = x[i], x[i]
		}
		y := FFTNWith(d, []int{n, 2}, Options{FMA: true})
		for i := range x {
			if cmplx.Abs(y[2*i]-2*want[i]) > 1e-8*float64(n) || cmplx.Abs(y[2*i+1]) > 1e-8*float64(n) {
				t.Fatalf("n=%d: FMA strip axis wrong", n)
			}
		}
	}
}

// fmaNaive is the O(n²) DFT with its angles reduced exactly (k·j mod n) and
// its sums compensated, accurate to about one rounding at the lengths here.
func fmaNaive(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	cs, sn := make([]float64, n), make([]float64, n)
	for m := range n {
		sn[m], cs[m] = math.Sincos(-2 * math.Pi * float64(m) / float64(n))
	}
	for k := range n {
		var sr, si, cr, ci float64
		for j, v := range x {
			s, c := sn[k*j%n], cs[k*j%n]
			tr := real(v)*c - imag(v)*s
			ti := real(v)*s + imag(v)*c
			// Kahan summation of both parts.
			yr, yi := tr-cr, ti-ci
			nr, ni := sr+yr, si+yi
			cr, ci = (nr-sr)-yr, (ni-si)-yi
			sr, si = nr, ni
		}
		out[k] = complex(sr, si)
	}
	return out
}

// TestFMAEveryRadix: plans that force each radix with an FMA twin first,
// last and between, every stage marked, transform correctly, differ from
// the default plan and are as accurate.
func TestFMAEveryRadix(t *testing.T) {
	forceFMA(t)
	for _, f := range [][]int{{9, 4}, {4, 9}, {12, 3}, {3, 12}, {16, 4}, {4, 16}, {16}, {10, 2}, {2, 10}, {15, 4}, {4, 15}, {20, 3},
		{3, 20}, {2, 8, 3}, {8, 8}, {5, 5, 4}, {9, 16}, {16, 9, 2}, {12, 12}, {7, 8}, {8, 11}} {
		n := factorProduct(f)
		p, d := newSKPlanFactorsOpt(n, f, true), newSKPlanFactors(n, f)
		for _, st := range p.stages {
			if st.fma != fmaRadix(st.r) || st.split != 0 && !fmaSplit {
				t.Fatalf("%v: stage of radix %d marked %v", f, st.r, st.fma)
			}
		}
		// Each interleaved stage alone: its kernel (or Go pass) against
		// passScalarFMA, the reference, which every radix then runs here.
		for k := range p.stages {
			st := &p.stages[k]
			if st.split != 0 || !st.fma {
				continue
			}
			in := fmaSignal(n, uint64(k)+5)
			a, b := make([]complex128, n), make([]complex128, n)
			st.pass(a, in, true)
			st.passScalarFMA(b, in, st.l1, true)
			if !sameAll(a, b) {
				t.Fatalf("%v stage %d: the pass differs from passScalarFMA", f, k)
			}
		}
		x := fmaSignal(n, uint64(n)+3)
		want := fmaNaive(x)
		for _, inverse := range []bool{false, true} {
			got, def := make([]complex128, n), make([]complex128, n)
			p.transform(got, x, inverse)
			d.transform(def, x, inverse)
			if inverse {
				for i := range want {
					want[i] = 0
				}
				continue
			}
			if e, ed := relRMS(got, want), relRMS(def, want); e > 1e-14 || e > 2*ed+1e-16 {
				t.Fatalf("%v: FMA error %.3g, default %.3g", f, e, ed)
			}
		}
	}
}

// TestFMACascade: an FMA plan on the blocked schedule (its strided pair,
// its passes over l1 blocks, its final runs) is bit-identical to the same
// FMA plan breadth first, on the kernels and on the Go passes.
func TestFMACascade(t *testing.T) {
	forceFMA(t)
	for _, n := range []int{1 << 14, 1 << 15, 1 << 16} {
		f := skFactorize(n)
		save := cascadeMin
		cascadeMin = 1
		blocked := newSKPlanFactorsOpt(n, f, true)
		cascadeMin = 0
		plain := newSKPlanFactorsOpt(n, f, true)
		cascadeMin = save
		if blocked.cascT == 0 {
			t.Fatalf("n=%d: no blocked schedule", n)
		}
		for s, x := range cascadeSignals(n) {
			for _, inverse := range []bool{false, true} {
				a, b := make([]complex128, n), make([]complex128, n)
				blocked.transform(a, x, inverse)
				plain.transform(b, x, inverse)
				if !sameAll(a, b) {
					t.Fatalf("n=%d signal %d inverse=%v: FMA blocked differs from FMA breadth first", n, s, inverse)
				}
			}
		}
		for k := range blocked.stages {
			st := &blocked.stages[k]
			// The Go paths of the stage's pieces, against its whole Go pass
			// (an FMA plan's stage may not fuse: fmaTable).
			x := fmaSignal(n, 5)
			want := make([]complex128, n)
			if st.fma {
				st.passScalarFMA(want, x, st.l1, false)
			} else {
				st.passScalarL(want, x, st.l1, false)
			}
			got := make([]complex128, n)
			if st.ido > 1 {
				st.cascadeStridedGo(got, x, st.ido, st.l1, st.ido, st.r*st.ido, st.l1*st.ido, st.ido, 0, 0, false)
			} else {
				st.cascadeLastGo(got, x, st.l1, st.l1, st.l1, false)
			}
			if !sameAll(got, want) {
				t.Fatalf("n=%d stage %d: the blocked schedule's Go pass differs", n, k)
			}
		}
	}
}

// TestFMARowsAndRotations: an FMA plan's large-array rows (clRowPlanF) and,
// the strips are the FMA plan's lines bit for bit, and arm64's rotating
// passes, where they run, as accurate.
func TestFMARowsAndRotations(t *testing.T) {
	forceFMA(t)
	defer func(m map[int][]int, v int) { clRowOrder, clRowMin = m, v }(clRowOrder, clRowMin)
	defer func(v, s bool) { armrealRotateND, stripAxes = v, s }(armrealRotateND, stripAxes)
	clRowOrder, clRowMin = map[int][]int{64: {2, 4, 8}}, 0
	p := clRowPlanF(64, 64*64, true)
	if p == nil || !p.sk.fma || clRowPlanF(64, 64*64, true) != p || clRowPlanF(64, 64*64, false) == p {
		t.Fatal("clRowPlanF: FMA row plans")
	}
	sh := []int{24, 64}
	x := fmaSignal(24*64, 13)
	var ref []complex128
	for _, c := range []struct{ rot, strips bool }{{false, false}, {false, true}, {true, true}} {
		armrealRotateND, stripAxes = c.rot, c.strips
		got := newPlanNF(sh, true).FFT(make([]complex128, len(x)), x)
		switch {
		case ref == nil:
			ref = got
		case !c.rot && !sameAll(got, ref):
			t.Fatalf("strips: an FMA N-D plan differs from its lines")
		case relRMS(got, ref) > 1e-15:
			// arm64's rotating passes are accurate, not bit-identical to the
			// rows and strips, in the default plans too.
			t.Fatalf("rotations: an FMA N-D plan is off by %.3g", relRMS(got, ref))
		}
	}
	// A stage run over many blocks, as the rotating passes run it.
	st := &newSKPlanFactorsOpt(32, []int{4, 8}, true).stages[0]
	in := fmaSignal(32*6, 17)
	a, b := make([]complex128, len(in)), make([]complex128, len(in))
	st.armrealPassL(a, in, 6*st.l1, false)
	st.passScalarFMA(b, in, 6*st.l1, false)
	if !sameAll(a, b) {
		t.Fatal("armrealPassL of an FMA stage differs from its Go pass")
	}
}

// TestFMASubPlansAreFMA: an FMA plan's convolution (Rader, Bluestein), and
// the strip axes that take a factorization of their own (intelStripOrder,
// cachedPlanNo16F), run on FMA plans when they transform, not only when they
// are built.
func TestFMASubPlansAreFMA(t *testing.T) {
	forceFMA(t)
	for _, n := range []int{1009, 10007} {
		p := newPlanF(n, true)
		q := n - 1
		if p.bluestein != nil {
			q = p.bluestein.m
		}
		forgetFMAPlans()
		p.FFT(make([]complex128, n), fmaSignal(n, 1))
		planMu.Lock()
		_, ok := planCacheFMA[q]
		planMu.Unlock()
		if !ok {
			t.Fatalf("n=%d: the transform did not run the FMA convolution plan of %d points", n, q)
		}
	}
	defer func(m map[int]bool) { intelStripOrder = m }(intelStripOrder)
	intelStripOrder = map[int]bool{48: true}
	if a := newPlanNF([]int{48, 8}, true).axes[0]; a == cachedPlanF(48, true) || a.sk == nil || !a.sk.fma {
		t.Fatal("a strip axis of its own factorization is not an FMA plan")
	}
}

// TestFMARealAndSubsetComposition: RFFTNWith and FFTNWith over a subset of
// axes are, bit for bit, the FMA plans they are built from.
func TestFMARealAndSubsetComposition(t *testing.T) {
	forceFMA(t)
	sh := []int{12, 20}
	r := fmaReal(240, 21)
	half := halfShape(sh, 1)
	want := make([]complex128, 12*half[1])
	realAxisForward(want, r, sh, half, 1, true)
	newPlanNAxes(half, []int{0}, true).transform(want, want, false)
	if !sameAll(RFFTNWith(r, sh, Options{FMA: true}), want) {
		t.Fatal("RFFTNWith is not its FMA plans' composition")
	}
	x := fmaSignal(240, 22)
	sub := make([]complex128, 240)
	pa := newPlanNAxes(sh, []int{1}, true)
	if pa.axes[1] != cachedPlanF(20, true) || pa.axes[0] != nil {
		t.Fatal("a plan over a subset of axes does not take the FMA axis plan")
	}
	pa.transform(sub, x, false)
	if !sameAll(FFTNWith(x, sh, Options{Axes: []int{1}, FMA: true}), sub) {
		t.Fatal("FFTNWith over a subset of axes is not the FMA plan")
	}
}
