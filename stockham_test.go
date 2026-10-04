package fft

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
	"unsafe"
)

// TestStockhamMatchesMixedRadix cross-checks the Stockham passes against the
// independent recursive mixed-radix engine at every smooth length up to 3000,
// in both directions and with dst aliasing src. That range reaches every pass
// (radix 2/3/4/5/7 and the general radix 11/13) at both ido == 1 and ido > 1,
// and plans with both an odd and an even number of passes.
func TestStockhamMatchesMixedRadix(t *testing.T) {
	for n := 2; n <= 3000; n++ {
		if !factorsAreSmall(n) {
			continue
		}
		x := cmplxSignal(n)
		sk, ct := newSKPlan(n), newCTPlan(n)
		tolN := 1e-12 * float64(n)
		for _, inverse := range []bool{false, true} {
			want := make([]complex128, n)
			ct.transform(want, x, inverse)
			got := make([]complex128, n)
			sk.transform(got, x, inverse)
			aliased := append([]complex128(nil), x...)
			sk.transform(aliased, aliased, inverse)
			for i := range want {
				if d := cmplx.Abs(got[i] - want[i]); d > tolN {
					t.Fatalf("n=%d inverse=%v index %d: Stockham %v vs mixed-radix %v", n, inverse, i, got[i], want[i])
				}
				if aliased[i] != got[i] {
					t.Fatalf("n=%d inverse=%v index %d: aliased %v vs %v", n, inverse, i, aliased[i], got[i])
				}
			}
		}
	}
}

// TestPooledConvolutionBufferIsCleared runs each convolution engine twice on
// one plan, first on a large input and then on a small one, and checks the
// second result against a fresh plan: the pooled buffer comes back dirty, so a
// zero pad that is not re-cleared would leak the first call into the second.
// Bluestein zero-pads its convolution: 10007 (a prime whose 10006 = 2·5003 is
// not smooth) and 1282 = 2·641 (a composite with a large prime factor) both
// route to it. Rader's cyclic convolution overwrites its whole buffer.
func TestPooledConvolutionBufferIsCleared(t *testing.T) {
	for _, n := range []int{1282, 10007} {
		dirty := NewPlan(n)
		big := make([]complex128, n)
		for i := range big {
			big[i] = complex(1e6, -1e6)
		}
		scratch := make([]complex128, n)
		dirty.FFT(scratch, big)

		x := cmplxSignal(n)
		got := dirty.FFT(make([]complex128, n), x)
		want := NewPlan(n).FFT(make([]complex128, n), x)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d index %d: reused plan %v vs fresh plan %v", n, i, got[i], want[i])
			}
		}
	}
}

// withPow2Route runs f with powers of two routed to the Stockham engine (on)
// or the iterative pow2 kernel, on a fresh plan cache, and restores both.
func withPow2Route(on bool, f func()) {
	planMu.Lock()
	saved, savedCache := pow2StockhamMax, planCache
	pow2StockhamMax, planCache = 0, map[int]*Plan{}
	if on {
		pow2StockhamMax = math.MaxInt
	}
	planMu.Unlock()
	defer func() {
		planMu.Lock()
		pow2StockhamMax, planCache = saved, savedCache
		planMu.Unlock()
	}()
	f()
}

// TestPow2RoutesAgree runs the complex and real transforms of powers of two
// down both routes — the Stockham engine and the iterative pow2 kernel (with
// its fused real packing) — and checks they agree. Whichever route this
// architecture uses by default, the other one stays tested here.
func TestPow2RoutesAgree(t *testing.T) {
	for _, n := range []int{2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 8192, 65536} {
		x := cmplxSignal(n)
		r := make([]float64, n)
		for i := range r {
			r[i] = real(x[i]) - 0.3*imag(x[i])
		}
		type out struct {
			f, i, a []complex128
			rf      []complex128
			ri      []float64
		}
		run := func() out {
			var o out
			p := NewPlan(n)
			o.f = p.FFT(make([]complex128, n), x)
			o.i = p.IFFT(make([]complex128, n), x)
			o.a = append([]complex128(nil), x...)
			p.FFT(o.a, o.a) // in place
			rp := NewRealPlan(n)
			o.rf = rp.RFFT(make([]complex128, n/2+1), r)
			o.ri = rp.IRFFT(make([]float64, n), o.rf)
			return o
		}
		var sk, it out
		withPow2Route(true, func() { sk = run() })
		withPow2Route(false, func() { it = run() })
		tol := 1e-12 * float64(n)
		for k := range sk.f {
			if cmplx.Abs(sk.f[k]-it.f[k]) > tol || cmplx.Abs(sk.i[k]-it.i[k]) > tol ||
				sk.a[k] != sk.f[k] || it.a[k] != it.f[k] {
				t.Fatalf("n=%d index %d: complex routes disagree", n, k)
			}
		}
		for k := range sk.rf {
			if cmplx.Abs(sk.rf[k]-it.rf[k]) > tol {
				t.Fatalf("n=%d bin %d: RFFT routes disagree: %v vs %v", n, k, sk.rf[k], it.rf[k])
			}
		}
		for k := range sk.ri {
			if math.Abs(sk.ri[k]-it.ri[k]) > tol || math.Abs(sk.ri[k]-r[k]) > tol {
				t.Fatalf("n=%d index %d: IRFFT routes disagree or do not round-trip", n, k)
			}
		}
	}
}

// TestAsComplexLayout pins the memory layout asComplex relies on: a complex128
// is its real float64 followed by its imaginary float64.
func TestAsComplexLayout(t *testing.T) {
	f := []float64{1, 2, 3, 4}
	c := asComplex(f)
	if len(c) != 2 || c[0] != complex(1, 2) || c[1] != complex(3, 4) {
		t.Fatalf("asComplex(%v) = %v", f, c)
	}
	c[1] = complex(5, 6)
	if f[2] != 5 || f[3] != 6 {
		t.Fatalf("a write through the view did not land in the float64s: %v", f)
	}
	if asComplex(nil) != nil {
		t.Fatal("asComplex(nil) != nil")
	}
}

// TestPow2Radices pins both ways an odd power of two can finish, whichever one
// this architecture uses (route_*.go), and that each product is 2^e.
func TestPow2Radices(t *testing.T) {
	cases := []struct {
		e         int
		oneRadix8 bool
		want      []int
	}{
		{1, true, []int{2}}, {2, true, []int{4}}, {7, true, []int{8, 4, 4}}, {9, true, []int{8, 4, 4, 4}},
		{8, true, []int{4, 4, 4, 4}}, {7, false, []int{4, 4, 4, 2}}, {9, false, []int{4, 4, 4, 4, 2}},
	}
	for _, c := range cases {
		got := pow2Radices(c.e, c.oneRadix8)
		if fmt.Sprint(got) != fmt.Sprint(c.want) {
			t.Errorf("pow2Radices(%d, %v) = %v, want %v", c.e, c.oneRadix8, got, c.want)
		}
		prod := 1
		for _, r := range got {
			prod *= r
		}
		if prod != 1<<c.e {
			t.Errorf("pow2Radices(%d, %v) multiplies to %d", c.e, c.oneRadix8, prod)
		}
	}
}

func TestWide512(t *testing.T) {
	for n, want := range map[int]bool{128: false, 256: true, 4096: true, 1 << 20: true, 1000: false, 768: false, 20160: false} {
		if got := wide512(n); got != want {
			t.Errorf("wide512(%d) = %v, want %v", n, got, want)
		}
	}
}

// TestScratchSitsOffTheSets checks offTheSets for every 16-byte alignment of
// dst against the scratch buffer: the window is n points inside the buffer
// and starts setGap bytes past dst, modulo 4 KB (within the 16-byte rounding
// of a complex128 index).
func TestScratchSitsOffTheSets(t *testing.T) {
	const n = 1024
	buf := make([]complex128, n+setSpan)
	backing := make([]complex128, n+setSpan)
	for k := 0; k < setSpan; k++ {
		dst := backing[k : k+n]
		w := offTheSets(buf, dst, n)
		if len(w) != n {
			t.Fatalf("dst offset %d: window length %d, want %d", k, len(w), n)
		}
		lo := uintptr(unsafe.Pointer(unsafe.SliceData(buf)))
		at := uintptr(unsafe.Pointer(unsafe.SliceData(w)))
		if at < lo || at+16*n > lo+16*uintptr(len(buf)) {
			t.Fatalf("dst offset %d: window outside the buffer", k)
		}
		gap := (at - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))) & 4095
		if gap > setGap || setGap-gap >= 16 {
			t.Fatalf("dst offset %d: window starts %d bytes past dst mod 4 KB, want %d", k, gap, setGap)
		}
	}
}

// TestSkFactorizeAroundR8Max takes both of skFactorize's power-of-two rules
// whatever this machine's r8MaxPow2 is: with AVX-512 it is unbounded, so the
// rule above it would otherwise go untested (and uncovered) there.
func TestSkFactorizeAroundR8Max(t *testing.T) {
	saved := r8MaxPow2
	defer func() { r8MaxPow2 = saved }()
	const e = 13
	r8MaxPow2 = 1 << e
	if got, want := fmt.Sprint(skFactorize(1<<e)), fmt.Sprint(radix8Maximal(e)); got != want {
		t.Errorf("at r8MaxPow2: skFactorize(2^%d) = %s, want radix 8 as far as it goes %s", e, got, want)
	}
	r8MaxPow2 = 1<<e - 1
	if got, want := fmt.Sprint(skFactorize(1<<e)), fmt.Sprint(pow2Radices(e, 1<<e <= pow2OneRadix8Max)); got != want {
		t.Errorf("above r8MaxPow2: skFactorize(2^%d) = %s, want %s", e, got, want)
	}
	// An odd factor keeps radix 8 as far as it goes for the power of two.
	if got, want := fmt.Sprint(skFactorize(3<<e)), fmt.Sprint(append(radix8Maximal(e), 3)); got != want {
		t.Errorf("skFactorize(3·2^%d) = %s, want %s", e, got, want)
	}
}

func TestTakesGap(t *testing.T) {
	for n, want := range map[int]bool{512: false, 1000: false, 1024: true, 1536: false, 1 << 20: true} {
		if got := takesGap(n); got != want {
			t.Errorf("takesGap(%d) = %v, want %v", n, got, want)
		}
	}
}
