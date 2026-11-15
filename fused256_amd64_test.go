package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The fused 256-point kernel (go-fft/fft#124) runs where AVX-512 F and DQ do
// and kernels.Fused256On is set (a Zen 4); it reports false elsewhere.

func skipNoFused(t testing.TB) {
	t.Helper()
	if !kernels.UseStockhamAVX512 {
		t.Skip("no AVX-512 here")
	}
}

// enableFused turns the kernel's own gate on for a test on any AVX-512 host
// (kernels.Fused256On is a Zen 4 choice) and restores it.
func enableFused(t testing.TB) {
	t.Helper()
	skipNoFused(t)
	isolatePlans(t)
	old := kernels.Fused256On
	kernels.Fused256On = true
	t.Cleanup(func() { kernels.Fused256On = old })
}

// runFused runs the kernel and fails the test if it refuses.
func runFused(t testing.TB, dst, src []complex128, inv bool) {
	t.Helper()
	if !kernels.Fused256((*[256]complex128)(dst), (*[256]complex128)(src), inv) {
		t.Fatal("the fused kernel refused with its gate on")
	}
}

// TestFused256Accuracy holds the kernel itself, both directions, to the stated
// bound (c2c: c·ε·log2 n relative RMS) against the double-double oracle, on
// random data, an impulse, a constant, a tone, and data spanning many binades.
func TestFused256Accuracy(t *testing.T) {
	enableFused(t)
	const n = 256
	inputs := map[string][]complex128{}
	for seed := uint64(1); seed <= 6; seed++ {
		inputs["random"+string(rune('0'+seed))] = accComplexF64(n, seed)
	}
	imp := make([]complex128, n)
	imp[1] = 1
	inputs["impulse"] = imp
	cst := make([]complex128, n)
	tone := make([]complex128, n)
	wide := make([]complex128, n)
	f := randSeq(n, 77)
	for i := range cst {
		cst[i] = complex(0.75, -0.25)
		tone[i] = complex(math.Cos(2*math.Pi*5*float64(i)/n), math.Sin(2*math.Pi*5*float64(i)/n))
		wide[i] = complex(f()*math.Exp2(float64(i%40-20)), f()*math.Exp2(float64((i*7)%40-20)))
	}
	inputs["constant"], inputs["tone"], inputs["wide"] = cst, tone, wide
	bound := fused256Bound()
	for name, x := range inputs {
		for _, inv := range []bool{false, true} {
			want := ddDFT(cddFromC(x), inv)
			got := make([]complex128, n)
			runFused(t, got, x, inv)
			if e := errC(got, want); e > bound {
				t.Errorf("%s inverse=%v: relative RMS error %.3g, above the bound %.3g", name, inv, e, bound)
			}
		}
	}
}

// fused256ImpulseCap is the largest absolute error of any output of any
// impulse, 1 or i at any of the 256 positions (the transform of an impulse is
// a row of roots of unity, so every table entry and every in-register constant
// is read by some impulse, with both parts nonzero). The worst observed, both
// directions, is about 2.7e-16 (TestFused256Impulses logs it); the cap is 1e-15,
// so a twiddle off by 1e-14 anywhere fails (the mutation test below proves it).
const fused256ImpulseCap = 1e-15

// worstImpulse is the largest absolute error over the 256 impulses, one
// direction, and where it was.
func worstImpulse(t testing.TB, inv bool) (worst float64, at [2]int) {
	t.Helper()
	const n = 256
	roots := ddRoots(n) // roots[k] = exp(-2πik/n)
	x := make([]complex128, n)
	got := make([]complex128, n)
	for j := 0; j < n; j++ {
		for _, unit := range []int{0, 1} { // the impulse 1, then i (so both parts of every intermediate are nonzero)
			x[j] = 1
			if unit == 1 {
				x[j] = 1i
			}
			runFused(t, got, x, inv)
			x[j] = 0
			for k := 0; k < n; k++ {
				w := roots[(j*k)%n]
				wr, wi := w.re, w.im
				if inv {
					wi = ddNeg(wi)
				}
				if unit == 1 { // times i: (wr + i wi) i = -wi + i wr
					wr, wi = ddNeg(wi), wr
				}
				dr := (real(got[k]) - wr.hi) - wr.lo
				di := (imag(got[k]) - wi.hi) - wi.lo
				if e := math.Max(math.Abs(dr), math.Abs(di)); e > worst {
					worst, at = e, [2]int{j, k}
				}
			}
		}
	}
	return
}

// TestFused256Impulses: every unit impulse, both directions, against the exact
// root of unity it must produce, to fused256ImpulseCap (as TestDCT4N64Impulses
// does for the DCT-IV): the random-data bound cannot see one table entry off by
// 1e-14, an impulse reads each entry directly.
func TestFused256Impulses(t *testing.T) {
	enableFused(t)
	for _, inv := range []bool{false, true} {
		w, at := worstImpulse(t, inv)
		t.Logf("inverse=%v: worst absolute error %.3g at impulse %d, output %d", inv, w, at[0], at[1])
		if w > fused256ImpulseCap {
			t.Errorf("inverse=%v: impulse %d, output %d: error %.3g above the cap %.3g", inv, at[0], at[1], w, fused256ImpulseCap)
		}
	}
}

// TestFused256ImpulseMutation: a table entry off by 1e-14, anywhere in either
// table, fails the impulse test; one off by an ulp passes it.
func TestFused256ImpulseMutation(t *testing.T) {
	enableFused(t)
	for _, inv := range []bool{false, true} {
		tab := kernels.Fused256Table(inv)
		for idx := 64; idx < len(tab); idx++ { // every entry of k1 >= 1 (row 0 is not read): real and imaginary parts, all lanes
			saved := tab[idx]
			tab[idx] = saved + 1e-14
			w, _ := worstImpulse(t, inv)
			tab[idx] = saved
			if w <= fused256ImpulseCap {
				t.Fatalf("inverse=%v: table[%d] off by 1e-14 gives worst error %.3g, under the cap", inv, idx, w)
			}
		}
		saved := tab[3]
		tab[3] = math.Nextafter(saved, 2)
		w, _ := worstImpulse(t, inv)
		tab[3] = saved
		if w > fused256ImpulseCap {
			t.Fatalf("inverse=%v: an ulp in table[3] fails the cap: %.3g", inv, w)
		}
	}
}

// TestFused256InPlace: dst may alias src (the kernel reads all of src before
// it writes dst).
func TestFused256InPlace(t *testing.T) {
	enableFused(t)
	x := accComplexF64(256, 3)
	want := make([]complex128, 256)
	runFused(t, want, x, false)
	buf := append([]complex128(nil), x...)
	runFused(t, buf, buf, false)
	for i := range buf {
		if buf[i] != want[i] {
			t.Fatalf("in place differs at %d: %v vs %v", i, buf[i], want[i])
		}
	}
}

// TestFused256Mutation shows what the bound sees in the kernel's twiddle table:
// a perturbation of one twiddle by 1e-10 turns the bound red, one by an ulp
// does not (it is below what the rounding does anyway), as for the oracle in
// TestAccuracyMutation.
func TestFused256Mutation(t *testing.T) {
	enableFused(t)
	const n = 256
	x := accComplexF64(n, 9)
	bound := fused256Bound()
	for _, inv := range []bool{false, true} {
		want := ddDFT(cddFromC(x), inv)
		tab := kernels.Fused256Table(inv)
		for _, idx := range []int{16*5 + 1, 16*40 + 3, 16*63 + 9} { // real part of a twiddle in three groups
			for _, m := range []struct {
				name   string
				delta  float64
				caught bool
			}{{"1 ulp", math.Nextafter(tab[idx], 2) - tab[idx], false}, {"1e-10", 1e-10, true}} {
				saved := tab[idx]
				tab[idx] += m.delta
				got := make([]complex128, n)
				runFused(t, got, x, inv)
				tab[idx] = saved
				if e := errC(got, want); (e > bound) != m.caught {
					t.Errorf("inverse=%v table[%d] %s: error %.3g against bound %.3g, caught = %v, want %v", inv, idx, m.name, e, bound, e > bound, m.caught)
				}
			}
		}
	}
}

// TestFused256GateIsLive: the kernel reads its gates at every call. With either
// Fused256On or UseStockhamAVX512 off it reports false and leaves dst alone, and
// a plan built while both were on still transforms correctly (its passes run), so
// a test that turns the AVX-512 flag off (fuzz_scalar_amd64_test.go) never
// reaches an AVX-512 instruction through it.
func TestFused256GateIsLive(t *testing.T) {
	enableFused(t)
	isolatePlans(t)
	defer func(e, a, d bool) { fused256Enabled, fmaAvailable, defaultFuses = e, a, d }(fused256Enabled, fmaAvailable, defaultFuses)
	fused256Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	p := NewPlan(256)
	x := accComplexF64(256, 31)
	want := ddDFT(cddFromC(x), false)
	sentinel := complex(7, 7)
	for name, off := range map[string]func() func(){
		"Fused256On":        func() func() { kernels.Fused256On = false; return func() { kernels.Fused256On = true } },
		"UseStockhamAVX512": func() func() { kernels.UseStockhamAVX512 = false; return func() { kernels.UseStockhamAVX512 = true } },
	} {
		restore := off()
		dst := make([]complex128, 256)
		for i := range dst {
			dst[i] = sentinel
		}
		if kernels.Fused256((*[256]complex128)(dst), (*[256]complex128)(x), false) {
			t.Errorf("%s off: the kernel ran", name)
		}
		for i := range dst {
			if dst[i] != sentinel {
				t.Fatalf("%s off: dst[%d] was written", name, i)
			}
		}
		if p.sk != nil && p.sk.f256 {
			if e := errC(p.FFT(make([]complex128, 256), x), want); e > fused256Bound() {
				t.Errorf("%s off: the plan's fallback passes give %.3g", name, e)
			}
		}
		restore()
	}
}

// TestFused256Routing: a fused default plan of 256 points takes the kernel;
// Exact and NoFMA plans, other lengths, and a plan built while the route is off
// never do.
func TestFused256Routing(t *testing.T) {
	isolatePlans(t)
	// The state the default arithmetic fuses in (other tests switch it).
	defer func(e, a, d bool, m int) {
		fused256Enabled, fmaAvailable, defaultFuses, pow2StockhamMax = e, a, d, m
	}(fused256Enabled, fmaAvailable, defaultFuses, pow2StockhamMax)
	fused256Enabled, fmaAvailable, defaultFuses, pow2StockhamMax = true, true, true, pow2StockhamMaxDefault()
	if !NewPlan(256).sk.f256 || !NewPlanWith(256, Options{FMA: true}).sk.f256 {
		t.Error("a default or FMA plan of 256 points does not take the fused kernel")
	}
	for name, o := range map[string]Options{"Exact": {Exact: true}, "NoFMA": {NoFMA: true}} {
		if NewPlanWith(256, o).sk.f256 {
			t.Errorf("%s plan takes the fused kernel", name)
		}
	}
	for _, n := range []int{128, 255, 512, 1024} {
		if p := NewPlan(n); p.sk != nil && p.sk.f256 {
			t.Errorf("plan of %d points takes the fused kernel", n)
		}
	}
	fused256Enabled = false
	if newPlanA(256, arithDefault).sk.f256 {
		t.Error("a plan built while the route is off takes the fused kernel")
	}
}

// TestFused256ForcedRoute runs a plan of 256 points with the route forced on:
// with the kernel's gate on (any AVX-512 host) the plan transforms through it,
// and without AVX-512 the kernel refuses and the plan's passes answer; either
// way within the bound, forward and inverse.
func TestFused256ForcedRoute(t *testing.T) {
	if len(kernels.Fused256Table(false)) != 1024 || len(kernels.Fused256Table(true)) != 1024 {
		t.Fatal("the twiddle tables are not 1024 floats")
	}
	if kernels.UseStockhamAVX512 {
		enableFused(t)
	}
	isolatePlans(t)
	defer func(e, a, d bool) { fused256Enabled, fmaAvailable, defaultFuses = e, a, d }(fused256Enabled, fmaAvailable, defaultFuses)
	fused256Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	p := NewPlan(256)
	x := accComplexF64(256, 21)
	for _, inv := range []bool{false, true} {
		got := make([]complex128, 256)
		if inv {
			p.IFFTNorm(got, x, NormForward)
		} else {
			p.FFT(got, x)
		}
		if e := errC(got, ddDFT(cddFromC(x), inv)); e > fused256Bound() {
			t.Errorf("inverse=%v: %.3g above the bound", inv, e)
		}
	}
}

// TestFused256Plan holds the public plan, forward and inverse (unnormalized and
// normalized), to the bound where the route is on, and Exact plans to the bound
// and to the v0.28.16 bits where they are not (TestExact* cover those bits).
func TestFused256Plan(t *testing.T) {
	skipNoFused(t)
	const n = 256
	x := accComplexF64(n, 5)
	bound := fused256Bound()
	for _, o := range []Options{{}, {FMA: true}, {Exact: true}, {NoFMA: true}} {
		p := NewPlanWith(n, o)
		got := p.FFT(make([]complex128, n), x)
		if e := errC(got, ddDFT(cddFromC(x), false)); e > bound {
			t.Errorf("%+v forward: %.3g above %.3g", o, e, bound)
		}
		got = p.IFFTNorm(make([]complex128, n), x, NormForward)
		if e := errC(got, ddDFT(cddFromC(x), true)); e > bound {
			t.Errorf("%+v inverse: %.3g above %.3g", o, e, bound)
		}
		back := p.IFFT(make([]complex128, n), p.FFT(make([]complex128, n), x))
		for i := range x {
			if d := back[i] - x[i]; math.Hypot(real(d), imag(d)) > 1e-14 {
				t.Fatalf("%+v round trip differs at %d by %g", o, i, d)
			}
		}
	}
}

func BenchmarkFused256(b *testing.B) {
	enableFused(b)
	const n = 256
	src := accComplexF64(n, 1)
	dst := make([]complex128, n)
	b.Run("kernel-fwd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			kernels.Fused256((*[256]complex128)(dst), (*[256]complex128)(src), false)
		}
	})
	for _, o := range []struct {
		name string
		o    Options
	}{{"default", Options{}}, {"exact", Options{Exact: true}}} {
		p := NewPlanWith(n, o.o)
		b.Run("plan-"+o.name+"-fwd", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
}

// TestFused256PlanNRows holds PlanN transforms whose rows are 256 points
// ({1,256}: only rows; {3,256}: rows and a strip axis of 3; {256,256}: rows and
// strips of 256), both directions (the inverse normalized), to the c2cN bound
// against the double-double N-D oracle. With AVX-512 the route is forced on, so
// the rows run the kernel (contiguousLines); elsewhere they run their passes.
func TestFused256PlanNRows(t *testing.T) {
	if kernels.UseStockhamAVX512 {
		enableFused(t)
	}
	isolatePlans(t)
	defer func(e, a, d bool) { fused256Enabled, fmaAvailable, defaultFuses = e, a, d }(fused256Enabled, fmaAvailable, defaultFuses)
	fused256Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	for _, shape := range [][]int{{1, 256}, {3, 256}, {256, 256}} {
		size := shape[0] * shape[1]
		x := accComplexF64(size, uint64(size))
		p := NewPlanN(shape...)
		bound := accBound["c2cN"] * eps64 * math.Log2(float64(size))
		for _, inv := range []bool{false, true} {
			got := make([]complex128, size)
			if inv {
				p.IFFT(got, x) // normalized, as the oracle's inverse is
			} else {
				p.FFT(got, x)
			}
			want := oFFTN(cddFromC(x), shape, allAxes(len(shape)), inv)
			if e := errC(got, want); e > bound {
				t.Errorf("shape %v inverse=%v: relative RMS error %.3g above the bound %.3g", shape, inv, e, bound)
			}
		}
	}
}
