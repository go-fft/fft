package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The fused 256-point kernel (go-fft/fft#124) runs wherever AVX-512 F and DQ
// do; the plan routes to it on a Zen 4 only (kernels.Fused256On).

func skipNoFused(t testing.TB) {
	t.Helper()
	if !kernels.UseStockhamAVX512 {
		t.Skip("no AVX-512 here")
	}
}

// TestFused256Accuracy holds the kernel itself, both directions, to the stated
// bound (c2c: c·ε·log2 n relative RMS) against the double-double oracle, on
// random data, an impulse, a constant, a tone, and data spanning many binades.
func TestFused256Accuracy(t *testing.T) {
	skipNoFused(t)
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
			kernels.Fused256(got, x, inv)
			if e := errC(got, want); e > bound {
				t.Errorf("%s inverse=%v: relative RMS error %.3g, above the bound %.3g", name, inv, e, bound)
			}
		}
	}
}

// TestFused256InPlace: dst may alias src (the kernel reads all of src before
// it writes dst).
func TestFused256InPlace(t *testing.T) {
	skipNoFused(t)
	x := accComplexF64(256, 3)
	want := make([]complex128, 256)
	kernels.Fused256(want, x, false)
	buf := append([]complex128(nil), x...)
	kernels.Fused256(buf, buf, false)
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
	skipNoFused(t)
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
				kernels.Fused256(got, x, inv)
				tab[idx] = saved
				if e := errC(got, want); (e > bound) != m.caught {
					t.Errorf("inverse=%v table[%d] %s: error %.3g against bound %.3g, caught = %v, want %v", inv, idx, m.name, e, bound, e > bound, m.caught)
				}
			}
		}
	}
}

// TestFused256Routing: a fused default plan of 256 points takes the kernel
// where it runs; Exact and NoFMA plans, other lengths, and a machine where it
// is off never do.
func TestFused256Routing(t *testing.T) {
	if !kernels.Fused256On {
		t.Skip("the fused kernel is off on this CPU (needs a Zen 4)")
	}
	isolatePlans(t)
	// The state the default arithmetic fuses in (other tests switch it).
	defer func(a, d bool, m int) { fmaAvailable, defaultFuses, pow2StockhamMax = a, d, m }(fmaAvailable, defaultFuses, pow2StockhamMax)
	fmaAvailable, defaultFuses, pow2StockhamMax = kernels.UseFMA, true, pow2StockhamMaxDefault()
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
	old := fused256Enabled
	fused256Enabled = false
	defer func() { fused256Enabled = old }()
	if newPlanA(256, arithDefault).sk.f256 {
		t.Error("a plan built while the route is off takes the fused kernel")
	}
}

// TestFused256ForcedRoute runs the plan's kernel case on any AVX-512 host (the
// route itself is a Zen 4 choice): the route switched on, a default plan of 256
// points transforms within the bound, forward and inverse.
func TestFused256ForcedRoute(t *testing.T) {
	skipNoFused(t)
	isolatePlans(t)
	defer func(e, a, d bool) { fused256Enabled, fmaAvailable, defaultFuses = e, a, d }(fused256Enabled, fmaAvailable, defaultFuses)
	fused256Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	p := NewPlan(256)
	x := accComplexF64(256, 21)
	for _, inv := range []bool{false, true} {
		if !p.sk.f256 { // no FMA3: the plan keeps its passes and this is nothing to test
			break
		}
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
	skipNoFused(b)
	const n = 256
	src := accComplexF64(n, 1)
	dst := make([]complex128, n)
	b.Run("kernel-fwd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			kernels.Fused256(dst, src, false)
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
