package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The fused 2048-point kernel (go-fft/fft#162) runs where AVX-512 F and DQ do and
// kernels.Fused2048On is set (a Zen 4); it reports false elsewhere.

// enableFused2048 turns the kernel's own gate on for a test on any AVX-512 host
// and restores it.
func enableFused2048(t testing.TB) {
	t.Helper()
	skipNoFused(t)
	isolatePlans(t)
	old := kernels.Fused2048On
	kernels.Fused2048On = true
	t.Cleanup(func() { kernels.Fused2048On = old })
}

// runFused2048 runs the kernel and fails the test if it refuses.
func runFused2048(t testing.TB, dst, src []complex128, inv bool) {
	t.Helper()
	if !kernels.Fused2048((*[2048]complex128)(dst), (*[2048]complex128)(src), inv) {
		t.Fatal("the fused 2048 kernel refused with its gate on")
	}
}

func fused2048Bound() float64 { return accBound["c2c"] * eps64 * math.Log2(2048) }

// TestFused2048Accuracy holds the kernel, both directions, to the c2c bound
// against the double-double oracle on random data, an impulse, a constant, a tone
// and data over many binades.
func TestFused2048Accuracy(t *testing.T) {
	enableFused2048(t)
	const n = 2048
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
	bound := fused2048Bound()
	for name, x := range inputs {
		for _, inv := range []bool{false, true} {
			want := ddDFT(cddFromC(x), inv)
			got := make([]complex128, n)
			runFused2048(t, got, x, inv)
			if e := errC(got, want); e > bound {
				t.Errorf("%s inverse=%v: relative RMS error %.3g, above the bound %.3g", name, inv, e, bound)
			}
		}
	}
}

// fused2048ImpulseCap is the largest absolute error of any output of any
// impulse, 1 or i at any of the 2048 positions (the transform of an impulse is a
// row of roots of unity, so every table entry and every in-register constant is
// read by some impulse with both parts nonzero). The worst observed, both
// directions, is 4.0e-16 (TestFused2048Impulses logs it); the cap is 2e-15, so a
// twiddle off by 1e-14 anywhere fails (the mutation test proves it).
const fused2048ImpulseCap = 2e-15

// worstImpulse2048 is the largest absolute error over the impulses at the given
// positions (nil: all 2048), 1 and i, one direction, and where it was.
func worstImpulse2048(t testing.TB, inv bool, positions []int) (worst float64, at [2]int) {
	t.Helper()
	const n = 2048
	roots := ddRoots(n) // roots[k] = exp(-2πik/n)
	if positions == nil {
		for j := 0; j < n; j++ {
			positions = append(positions, j)
		}
	}
	x := make([]complex128, n)
	got := make([]complex128, n)
	for _, j := range positions {
		for _, unit := range []int{0, 1} {
			x[j] = 1
			if unit == 1 {
				x[j] = 1i
			}
			runFused2048(t, got, x, inv)
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

// TestFused2048Impulses: every unit impulse (1 and i), both directions, against
// the exact root of unity it must produce, to fused2048ImpulseCap.
func TestFused2048Impulses(t *testing.T) {
	enableFused2048(t)
	for _, inv := range []bool{false, true} {
		w, at := worstImpulse2048(t, inv, nil)
		t.Logf("inverse=%v: worst absolute error %.3g at impulse %d, output %d", inv, w, at[0], at[1])
		if w > fused2048ImpulseCap {
			t.Errorf("inverse=%v: impulse %d, output %d: error %.3g above the cap %.3g", inv, at[0], at[1], w, fused2048ImpulseCap)
		}
	}
}

// TestFused2048ImpulseMutation: a table entry off by 1e-14 fails the impulse test
// at the columns that read it; one off by an ulp passes. Every 13th entry of each
// table is mutated (an entry (group g, k1, lane l) is read by the impulses at
// positions 64a + 4g + l), and the stride, coprime to 16, covers every entry kind
// (real and imaginary, even and odd copy).
func TestFused2048ImpulseMutation(t *testing.T) {
	enableFused2048(t)
	for _, inv := range []bool{false, true} {
		tab := kernels.Fused2048Table(inv)
		for idx := 0; idx < len(tab); idx += 13 {
			entry, within := idx/16, idx%16
			g := entry / 31
			lane := (within % 8) / 2
			var pos []int
			for a := 0; a < 32; a++ {
				pos = append(pos, 64*a+4*g+lane)
			}
			saved := tab[idx]
			tab[idx] = saved + 1e-14
			w, _ := worstImpulse2048(t, inv, pos)
			tab[idx] = saved
			if w <= fused2048ImpulseCap {
				t.Fatalf("inverse=%v: table[%d] off by 1e-14 gives worst error %.3g at its columns, under the cap", inv, idx, w)
			}
		}
		saved := tab[3]
		tab[3] = math.Nextafter(saved, 2)
		w, _ := worstImpulse2048(t, inv, []int{0, 1, 5, 64, 77})
		tab[3] = saved
		if w > fused2048ImpulseCap {
			t.Fatalf("inverse=%v: an ulp in table[3] fails the cap: %.3g", inv, w)
		}
	}
}

// TestFused2048InPlace: dst may alias src (pass 1 reads all of src before pass 2
// writes dst).
func TestFused2048InPlace(t *testing.T) {
	enableFused2048(t)
	x := accComplexF64(2048, 3)
	want := make([]complex128, 2048)
	runFused2048(t, want, x, false)
	buf := append([]complex128(nil), x...)
	runFused2048(t, buf, buf, false)
	for i := range buf {
		if buf[i] != want[i] {
			t.Fatalf("in place differs at %d: %v vs %v", i, buf[i], want[i])
		}
	}
}

// TestFused2048GateIsLive: the kernel reads its gates at every call, as the 256
// one does.
func TestFused2048GateIsLive(t *testing.T) {
	enableFused2048(t)
	isolatePlans(t)
	defer func(e, a, d bool) { fused2048Enabled, fmaAvailable, defaultFuses = e, a, d }(fused2048Enabled, fmaAvailable, defaultFuses)
	fused2048Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	p := NewPlan(2048)
	x := accComplexF64(2048, 31)
	want := ddDFT(cddFromC(x), false)
	sentinel := complex(7, 7)
	for name, off := range map[string]func() func(){
		"Fused2048On":       func() func() { kernels.Fused2048On = false; return func() { kernels.Fused2048On = true } },
		"UseStockhamAVX512": func() func() { kernels.UseStockhamAVX512 = false; return func() { kernels.UseStockhamAVX512 = true } },
	} {
		restore := off()
		dst := make([]complex128, 2048)
		for i := range dst {
			dst[i] = sentinel
		}
		if kernels.Fused2048((*[2048]complex128)(dst), (*[2048]complex128)(x), false) {
			t.Errorf("%s off: the kernel ran", name)
		}
		for i := range dst {
			if dst[i] != sentinel {
				t.Fatalf("%s off: dst[%d] was written", name, i)
			}
		}
		if p.sk != nil && p.sk.f2048 {
			if e := errC(p.FFT(make([]complex128, 2048), x), want); e > fused2048Bound() {
				t.Errorf("%s off: the plan's fallback passes give %.3g", name, e)
			}
		}
		restore()
	}
}

// TestFused2048Routing: a fused default plan of 2048 points takes the kernel;
// Exact and NoFMA plans, other lengths, and a plan built while the route is off
// never do.
func TestFused2048Routing(t *testing.T) {
	isolatePlans(t)
	defer func(e, a, d bool, m int) {
		fused2048Enabled, fmaAvailable, defaultFuses, pow2StockhamMax = e, a, d, m
	}(fused2048Enabled, fmaAvailable, defaultFuses, pow2StockhamMax)
	fused2048Enabled, fmaAvailable, defaultFuses, pow2StockhamMax = true, true, true, pow2StockhamMaxDefault()
	if !NewPlan(2048).sk.f2048 || !NewPlanWith(2048, Options{FMA: true}).sk.f2048 {
		t.Error("a default or FMA plan of 2048 points does not take the fused kernel")
	}
	for name, o := range map[string]Options{"Exact": {Exact: true}, "NoFMA": {NoFMA: true}} {
		if NewPlanWith(2048, o).sk.f2048 {
			t.Errorf("%s plan takes the fused kernel", name)
		}
	}
	for _, n := range []int{256, 1024, 2000, 4096} {
		if p := NewPlan(n); p.sk != nil && p.sk.f2048 {
			t.Errorf("plan of %d points takes the fused 2048 kernel", n)
		}
	}
	fused2048Enabled = false
	if newPlanA(2048, arithDefault).sk.f2048 {
		t.Error("a plan built while the route is off takes the fused kernel")
	}
}

// TestFused2048ForcedRoute runs a plan of 2048 points with the route forced on:
// through the kernel where its gate is on, and through the plan's passes where it
// refuses (no AVX-512), within the bound either way, forward and inverse.
func TestFused2048ForcedRoute(t *testing.T) {
	if len(kernels.Fused2048Table(false)) != 7936 || len(kernels.Fused2048Table(true)) != 7936 {
		t.Fatal("the twiddle tables are not 7936 floats")
	}
	if kernels.UseStockhamAVX512 {
		enableFused2048(t)
	}
	isolatePlans(t)
	defer func(e, a, d bool) { fused2048Enabled, fmaAvailable, defaultFuses = e, a, d }(fused2048Enabled, fmaAvailable, defaultFuses)
	fused2048Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	p := NewPlan(2048)
	x := accComplexF64(2048, 21)
	for _, inv := range []bool{false, true} {
		got := make([]complex128, 2048)
		if inv {
			p.IFFTNorm(got, x, NormForward)
		} else {
			p.FFT(got, x)
		}
		if e := errC(got, ddDFT(cddFromC(x), inv)); e > fused2048Bound() {
			t.Errorf("inverse=%v: %.3g above the bound", inv, e)
		}
	}
}

// TestFused2048Plan holds the public plan, in four arithmetics, to the bound.
func TestFused2048Plan(t *testing.T) {
	skipNoFused(t)
	const n = 2048
	x := accComplexF64(n, 5)
	bound := fused2048Bound()
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

// TestFused2048PlanNRows holds PlanN transforms whose rows are 2048 points
// ({1,2048}, {3,2048}, {8,2048}) to the c2cN bound against the double-double N-D
// oracle, both directions (the inverse normalized), with the route forced on so
// the rows run the kernel where the CPU allows it.
func TestFused2048PlanNRows(t *testing.T) {
	if kernels.UseStockhamAVX512 {
		enableFused2048(t)
	}
	isolatePlans(t)
	defer func(e, a, d bool) { fused2048Enabled, fmaAvailable, defaultFuses = e, a, d }(fused2048Enabled, fmaAvailable, defaultFuses)
	fused2048Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	for _, shape := range [][]int{{1, 2048}, {3, 2048}, {8, 2048}} {
		size := shape[0] * shape[1]
		x := accComplexF64(size, uint64(size))
		p := NewPlanN(shape...)
		bound := accBound["c2cN"] * eps64 * math.Log2(float64(size))
		for _, inv := range []bool{false, true} {
			got := make([]complex128, size)
			if inv {
				p.IFFT(got, x)
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

// TestFused2048RealPlan: the half-length complex plan of a real transform of 4096
// points is a Plan, so RFFT and IRFFT of 4096 take the kernel and stay within the
// real-transform bound against the oracle.
func TestFused2048RealPlan(t *testing.T) {
	if kernels.UseStockhamAVX512 {
		enableFused2048(t)
	}
	isolatePlans(t)
	defer func(e, a, d bool) { fused2048Enabled, fmaAvailable, defaultFuses = e, a, d }(fused2048Enabled, fmaAvailable, defaultFuses)
	fused2048Enabled, fmaAvailable, defaultFuses = true, kernels.UseFMA, true
	const n = 4096
	p := NewRealPlan(n)
	if p.half == nil || p.half.sk == nil || !p.half.sk.f2048 && kernels.UseStockhamAVX512 && kernels.Fused2048On && fmaAvailable {
		t.Fatal("the half plan of a real transform of 4096 points does not take the fused kernel")
	}
	x := accRealF64(n, 9)
	got := p.RFFT(make([]complex128, n/2+1), x)
	want := oRFFT(cddFromR(x))
	if e := errC(got, want); e > accBound["r2c"]*eps64*math.Log2(n) {
		t.Errorf("RFFT 4096: relative RMS error %.3g above the bound", e)
	}
	back := p.IRFFT(make([]float64, n), got)
	for i := range x {
		if math.Abs(back[i]-x[i]) > 1e-14 {
			t.Fatalf("real round trip differs at %d by %g", i, back[i]-x[i])
		}
	}
}

func BenchmarkFused2048(b *testing.B) {
	enableFused2048(b)
	const n = 2048
	src := accComplexF64(n, 1)
	dst := make([]complex128, n)
	b.Run("kernel-fwd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			kernels.Fused2048((*[2048]complex128)(dst), (*[2048]complex128)(src), false)
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
