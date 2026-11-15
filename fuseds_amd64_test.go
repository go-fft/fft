package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The fused complex64 kernels (go-fft/fft#162) run where AVX-512 F and DQ do and
// the kernel's own gate is set (an AMD Zen 4 or an Intel Cascade Lake); they
// report false elsewhere. Every test turns the gates on for itself, on any
// AVX-512 host, and restores them.

// fusedSCase is one kernel: its length, the call, its table, its gate and the
// A x B factorization it was generated with (x[B*a+b]).
type fusedSCase struct {
	n    int
	run  func(dst, src []complex64, inv bool) bool
	tab  func(inverse bool) []float32
	on   *bool
	a, b int
}

func fusedSCases() []fusedSCase {
	return []fusedSCase{
		{64, func(d, s []complex64, i bool) bool {
			return kernels.FusedS64((*[64]complex64)(d), (*[64]complex64)(s), i)
		}, kernels.FusedS64Table, &kernels.FusedS64On, 8, 8},
		{128, func(d, s []complex64, i bool) bool {
			return kernels.FusedS128((*[128]complex64)(d), (*[128]complex64)(s), i)
		}, kernels.FusedS128Table, &kernels.FusedS128On, 16, 8},
		{256, func(d, s []complex64, i bool) bool {
			return kernels.FusedS256((*[256]complex64)(d), (*[256]complex64)(s), i)
		}, kernels.FusedS256Table, &kernels.FusedS256On, 16, 16},
		{512, func(d, s []complex64, i bool) bool {
			return kernels.FusedS512((*[512]complex64)(d), (*[512]complex64)(s), i)
		}, kernels.FusedS512Table, &kernels.FusedS512On, 32, 16},
		{1024, func(d, s []complex64, i bool) bool {
			return kernels.FusedS1024((*[1024]complex64)(d), (*[1024]complex64)(s), i)
		}, kernels.FusedS1024Table, &kernels.FusedS1024On, 32, 32},
		{2048, func(d, s []complex64, i bool) bool {
			return kernels.FusedS2048((*[2048]complex64)(d), (*[2048]complex64)(s), i)
		}, kernels.FusedS2048Table, &kernels.FusedS2048On, 64, 32},
		{4096, func(d, s []complex64, i bool) bool {
			return kernels.FusedS4096((*[4096]complex64)(d), (*[4096]complex64)(s), i)
		}, kernels.FusedS4096Table, &kernels.FusedS4096On, 64, 64},
		{8192, func(d, s []complex64, i bool) bool {
			return kernels.FusedS8192((*[8192]complex64)(d), (*[8192]complex64)(s), i)
		}, kernels.FusedS8192Table, &kernels.FusedS8192On, 128, 64},
	}
}

// enableFusedS turns every complex64 kernel's gate (and the plans' route) on for
// a test on any AVX-512 host and restores them.
func enableFusedS(t testing.TB) {
	t.Helper()
	skipNoFused(t)
	isolatePlans(t)
	for _, c := range fusedSCases() {
		c, old, oldRoute := c, *c.on, fusedSEnabled[c.n]
		*c.on, fusedSEnabled[c.n] = true, true
		t.Cleanup(func() { *c.on, fusedSEnabled[c.n] = old, oldRoute })
	}
}

// runFusedS runs the kernel and fails the test if it refuses.
func runFusedS(t testing.TB, c fusedSCase, dst, src []complex64, inv bool) {
	t.Helper()
	if !c.run(dst, src, inv) {
		t.Fatalf("the fused complex64 %d kernel refused with its gate on", c.n)
	}
}

// fusedSBound is the c2c bound in single precision: c·eps32·log2 n, eps32 = 2^-23.
func fusedSBound(n int) float64 { return accBound["c2c"] * eps32 * math.Log2(float64(n)) }

func toC64Slice(x []complex128) []complex64 {
	out := make([]complex64, len(x))
	for i, v := range x {
		out[i] = complex(float32(real(v)), float32(imag(v)))
	}
	return out
}

// TestFusedSAccuracy holds every kernel, both directions, to the c2c bound
// against the double-double oracle on random data, an impulse, a constant, a tone
// and data over many binades (in float32 range).
func TestFusedSAccuracy(t *testing.T) {
	enableFusedS(t)
	for _, c := range fusedSCases() {
		n := c.n
		inputs := map[string][]complex64{}
		for seed := uint64(1); seed <= 4; seed++ {
			inputs["random"+string(rune('0'+seed))] = toC64Slice(accComplexF64(n, seed))
		}
		imp := make([]complex64, n)
		imp[1] = 1
		inputs["impulse"] = imp
		cst := make([]complex64, n)
		tone := make([]complex64, n)
		wide := make([]complex64, n)
		f := randSeq(n, 77)
		for i := range cst {
			cst[i] = complex(0.75, -0.25)
			tone[i] = complex(float32(math.Cos(2*math.Pi*5*float64(i)/float64(n))), float32(math.Sin(2*math.Pi*5*float64(i)/float64(n))))
			wide[i] = complex(float32(f()*math.Exp2(float64(i%24-12))), float32(f()*math.Exp2(float64((i*7)%24-12))))
		}
		inputs["constant"], inputs["tone"], inputs["wide"] = cst, tone, wide
		bound := fusedSBound(n)
		for name, x := range inputs {
			for _, inv := range []bool{false, true} {
				want := ddDFT(cddFromC(x), inv)
				got := make([]complex64, n)
				runFusedS(t, c, got, x, inv)
				if e := errC(got, want); e > bound {
					t.Errorf("n=%d %s inverse=%v: relative RMS error %.3g, above the bound %.3g", n, name, inv, e, bound)
				}
			}
		}
	}
}

// fusedSImpulseCap is the largest absolute error of any output of any impulse
// (1 or i) allowed: 2^-24 times (2 + log2 n), 8 to 16 units, 3.5 to 10 times the
// worst observed (the transform of an
// impulse is a row of roots of unity, so every table entry and every in-register
// constant is read by some impulse with both parts nonzero). The worst observed
// is a fraction of that (TestFusedSImpulses logs it); a twiddle off by 1e-4
// anywhere fails it (the mutation test proves it).
func fusedSImpulseCap(n int) float64 { return math.Ldexp(1, -24) * (2 + math.Log2(float64(n))) }

// fusedSPositions are the impulse positions tested: every one up to 4096
// points; above, every column b with the rows a = 0, 1, A/2+1 and A-1 (each
// table entry and constant is read by impulses of every column).
func fusedSPositions(c fusedSCase) []int {
	if c.n <= 4096 {
		return nil
	}
	var pos []int
	for _, a := range []int{0, 1, c.a/2 + 1, c.a - 1} {
		for b := 0; b < c.b; b++ {
			pos = append(pos, c.b*a+b)
		}
	}
	return pos
}

// worstImpulseS is the largest absolute error over the impulses at positions (nil:
// all n), 1 and i, one direction, and where it was.
func worstImpulseS(t testing.TB, c fusedSCase, inv bool, positions []int) (worst float64, at [2]int) {
	t.Helper()
	n := c.n
	roots := ddRoots(n) // roots[k] = exp(-2πik/n)
	if positions == nil {
		for j := 0; j < n; j++ {
			positions = append(positions, j)
		}
	}
	x := make([]complex64, n)
	got := make([]complex64, n)
	for _, j := range positions {
		for _, unit := range []int{0, 1} {
			x[j] = 1
			if unit == 1 {
				x[j] = 1i
			}
			runFusedS(t, c, got, x, inv)
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
				dr := (float64(real(got[k])) - wr.hi) - wr.lo
				di := (float64(imag(got[k])) - wi.hi) - wi.lo
				if e := math.Max(math.Abs(dr), math.Abs(di)); e > worst {
					worst, at = e, [2]int{j, k}
				}
			}
		}
	}
	return
}

// TestFusedSImpulses: every unit impulse (1 and i), both directions, against the
// exact root of unity it must produce, to fusedSImpulseCap.
func TestFusedSImpulses(t *testing.T) {
	enableFusedS(t)
	for _, c := range fusedSCases() {
		for _, inv := range []bool{false, true} {
			w, at := worstImpulseS(t, c, inv, fusedSPositions(c))
			t.Logf("n=%d inverse=%v: worst absolute error %.3g (cap %.3g) at impulse %d, output %d", c.n, inv, w, fusedSImpulseCap(c.n), at[0], at[1])
			if w > fusedSImpulseCap(c.n) {
				t.Errorf("n=%d inverse=%v: impulse %d, output %d: error %.3g above the cap %.3g", c.n, inv, at[0], at[1], w, fusedSImpulseCap(c.n))
			}
		}
	}
}

// TestFusedSImpulseMutation: a table entry off by 1e-4 fails the impulse test at
// the columns that read it; one off by an ulp passes. About 300 entries of each
// table are mutated (an entry (group g, k1, lane l) is read by the impulses at
// positions B*a + 8g + l), the stride being odd, so every entry kind (real and
// imaginary, even and odd copy, every lane) is covered.
func TestFusedSImpulseMutation(t *testing.T) {
	enableFusedS(t)
	for _, c := range fusedSCases() {
		for _, inv := range []bool{false, true} {
			tab := c.tab(inv)
			stride := len(tab)/300 | 1
			for idx := 0; idx < len(tab); idx += stride {
				entry, within := idx/32, idx%32
				g := entry / (c.a - 1)
				lane := (within % 16) / 2
				b := 8*g + lane
				pos := []int{b, c.b*(c.a/2+1) + b}
				saved := tab[idx]
				tab[idx] = saved + 1e-4
				w, _ := worstImpulseS(t, c, inv, pos)
				tab[idx] = saved
				if w <= fusedSImpulseCap(c.n) {
					t.Fatalf("n=%d inverse=%v: table[%d] off by 1e-4 gives worst error %.3g at its columns, under the cap %.3g", c.n, inv, idx, w, fusedSImpulseCap(c.n))
				}
			}
			saved := tab[3]
			tab[3] = math.Nextafter32(saved, 2)
			w, _ := worstImpulseS(t, c, inv, []int{0, 1, 5, c.b, c.b + 3})
			tab[3] = saved
			if w > fusedSImpulseCap(c.n) {
				t.Fatalf("n=%d inverse=%v: an ulp in table[3] fails the cap: %.3g", c.n, inv, w)
			}
		}
	}
}

// TestFusedSInPlace: dst may alias src (pass 1 reads all of src before pass 2
// writes dst).
func TestFusedSInPlace(t *testing.T) {
	enableFusedS(t)
	for _, c := range fusedSCases() {
		x := toC64Slice(accComplexF64(c.n, 3))
		want := make([]complex64, c.n)
		runFusedS(t, c, want, x, false)
		buf := append([]complex64(nil), x...)
		runFusedS(t, c, buf, buf, false)
		for i := range buf {
			if buf[i] != want[i] {
				t.Fatalf("n=%d: in place differs at %d: %v vs %v", c.n, i, buf[i], want[i])
			}
		}
	}
}

// TestFusedSGateIsLive: every kernel reads its gates at every call and writes
// nothing when one is off.
func TestFusedSGateIsLive(t *testing.T) {
	enableFusedS(t)
	isolatePlans(t)
	sentinel := complex64(complex(7, 7))
	for _, c := range fusedSCases() {
		x := toC64Slice(accComplexF64(c.n, 31))
		for name, off := range map[string]func() func(){
			"gate":              func() func() { *c.on = false; return func() { *c.on = true } },
			"UseStockhamAVX512": func() func() { kernels.UseStockhamAVX512 = false; return func() { kernels.UseStockhamAVX512 = true } },
		} {
			restore := off()
			dst := make([]complex64, c.n)
			for i := range dst {
				dst[i] = sentinel
			}
			if c.run(dst, x, false) {
				t.Errorf("n=%d %s off: the kernel ran", c.n, name)
			}
			for i := range dst {
				if dst[i] != sentinel {
					t.Fatalf("n=%d %s off: dst[%d] was written", c.n, name, i)
				}
			}
			// the plan falls back to its passes within the bound
			p := NewPlan32(c.n)
			want := ddDFT(cddFromC(x), false)
			if e := errC(p.FFT(make([]complex64, c.n), x), want); e > fusedSBound(c.n) {
				t.Errorf("n=%d %s off: the plan's fallback passes give %.3g", c.n, name, e)
			}
			restore()
		}
	}
}

// TestFusedSRouting: the public 1-D plans of the kernels' lengths take them; the
// plans under the N-D, 2-D real, DCT, DST and MDCT transforms, the Exact calls
// and a plan built while the route is off never do; nor does any other length.
func TestFusedSRouting(t *testing.T) {
	enableFusedS(t)
	isolatePlans(t)
	for _, c := range fusedSCases() {
		if p := NewPlan32(c.n); p.fs != c.n {
			t.Errorf("NewPlan32(%d) does not take the fused kernel", c.n)
		}
		if p := cachedPlan32F(c.n, true); p.fs != c.n {
			t.Errorf("the fused cached plan of %d does not take the fused kernel", c.n)
		}
		if p := cachedPlan32(c.n); p.fs != 0 {
			t.Errorf("the plain cached plan of %d takes the fused kernel", c.n)
		}
		if p := newPlan32(c.n, false); p.fs != 0 {
			t.Errorf("newPlan32(%d, false) takes the fused kernel", c.n)
		}
		if f := NewPlanN32(c.n, 3).axes[0]; f.fs != 0 {
			t.Errorf("a PlanN32 axis of %d takes the fused kernel", c.n)
		}
		fusedSEnabled[c.n] = false
		if NewPlan32(c.n).fs != 0 {
			t.Errorf("a plan of %d built while the route is off takes the fused kernel", c.n)
		}
		fusedSEnabled[c.n] = true
	}
	for _, n := range []int{2, 100, 1000, 2000, 4097, 32768, 65536} {
		if NewPlan32(n).fs != 0 {
			t.Errorf("a plan of %d points takes a fused kernel", n)
		}
	}
	// the real plans: the half plan of 2n points
	if p := NewRealPlan32(8192); p.half == nil || p.half.fs != 4096 {
		t.Error("RealPlan32 of 8192 points does not take the fused 4096 kernel")
	}
	if p := cachedRealPlan32(8192); p.half.fs != 0 {
		t.Error("the plain cached real plan of 8192 points takes the fused kernel")
	}
}

// TestFusedSExactAndInternalPlansKeepTheirBits: Options.Exact calls, the DCT32
// and MDCT32 plans and the N-D plans (golden digests pin their bits, whatever the
// CPU) are the unfused passes' bits, kernels on or off.
func TestFusedSExactAndInternalPlansKeepTheirBits(t *testing.T) {
	enableFusedS(t)
	for _, c := range fusedSCases() {
		x := toC64Slice(accComplexF64(c.n, 9))
		plain := make([]complex64, c.n)
		cachedPlan32(c.n).FFT(plain, x)
		got := FFT32With(x, Options{Exact: true})
		for i := range got {
			if got[i] != plain[i] {
				t.Fatalf("n=%d: FFT32With(Exact) differs from the passes at %d", c.n, i)
			}
		}
		fused := FFT32(x)
		diff := false
		for i := range fused {
			diff = diff || fused[i] != plain[i]
		}
		if !diff {
			t.Errorf("n=%d: the fused and the plain transform agree to the bit: the route did not run", c.n)
		}
		r := make([]float32, c.n)
		for i := range r {
			r[i] = float32(real(x[i]))
		}
		want := RFFT32With(r, Options{Exact: true})
		plainReal := cachedRealPlan32(c.n).RFFT(make([]complex64, c.n/2+1), r)
		for i := range want {
			if want[i] != plainReal[i] {
				t.Fatalf("n=%d: RFFT32With(Exact) differs from the passes at %d", c.n, i)
			}
		}
	}
}

// TestFusedSRealPlans: the real transforms whose half plan is a kernel, forward
// and inverse, within the bound against the double-double oracle.
func TestFusedSRealPlans(t *testing.T) {
	enableFusedS(t)
	for _, c := range fusedSCases() {
		n := 2 * c.n
		if n > 32768 {
			continue
		}
		xr := make([]float32, n)
		x := make([]complex128, n)
		for i, v := range accComplexF64(n, 5) {
			xr[i] = float32(real(v))
			x[i] = complex(float64(xr[i]), 0)
		}
		want := ddDFT(cddFromC(x), false)[:n/2+1]
		p := NewRealPlan32(n)
		if p.half == nil || p.half.fs != c.n {
			t.Fatalf("RealPlan32(%d): its half plan does not take the kernel", n)
		}
		spec := p.RFFT(make([]complex64, n/2+1), xr)
		if e := errC(spec, want); e > fusedSBound(n) {
			t.Errorf("RFFT32 n=%d: %.3g above the bound %.3g", n, e, fusedSBound(n))
		}
		back := p.IRFFT(make([]float32, n), spec)
		var num, den float64
		for i := range back {
			d := float64(back[i]) - float64(xr[i])
			num += d * d
			den += float64(xr[i]) * float64(xr[i])
		}
		if e := math.Sqrt(num / den); e > 2*fusedSBound(n) {
			t.Errorf("IRFFT32(RFFT32) n=%d: round trip error %.3g above %.3g", n, e, 2*fusedSBound(n))
		}
	}
}
