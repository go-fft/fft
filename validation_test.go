package fft

import (
	"math"
	"math/cmplx"
	"strings"
	"testing"
)

// mustPanicWith runs f and requires a panic whose message starts with prefix.
func mustPanicWith(t *testing.T, name, prefix string, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		msg, _ := r.(string)
		if r == nil || !strings.HasPrefix(msg, prefix) {
			t.Errorf("%s: recovered %v, want a panic starting %q", name, r, prefix)
		}
	}()
	f()
}

// TestShapeOverflowPanics: a shape whose product overflows int used to wrap
// to 0 = len(nil), pass the length check, and end in an unrecoverable
// "fatal error: out of memory" while planning its 2^32-point axes. It must be
// an ordinary panic, before anything is allocated.
func TestShapeOverflowPanics(t *testing.T) {
	const want = "fft: shape product overflows int"
	huge := []int{1 << 32, 1 << 32}
	mustPanicWith(t, "FFTN", want, func() { FFTN(nil, huge) })
	mustPanicWith(t, "IFFTN", want, func() { IFFTN(nil, huge) })
	mustPanicWith(t, "FFT2", want, func() { FFT2(nil, [2]int{1 << 32, 1 << 32}) })
	mustPanicWith(t, "IFFT2", want, func() { IFFT2(nil, [2]int{1 << 32, 1 << 32}) })
	mustPanicWith(t, "RFFT2", want, func() { RFFT2(nil, [2]int{1 << 32, 1 << 32}) })
	mustPanicWith(t, "IRFFT2", want, func() { IRFFT2(nil, [2]int{1 << 32, 1 << 32}) })
	mustPanicWith(t, "NewPlanN", want, func() { NewPlanN(1<<62, 4) })
	mustPanicWith(t, "NewRealPlan2", want, func() { NewRealPlan2(1<<62, 4) })
	mustPanicWith(t, "FFTN three axes", want, func() { FFTN(nil, []int{1 << 21, 1 << 21, 1 << 21}) })
	mustPanicWith(t, "FFTN zero", "fft: shape lengths must be positive", func() { FFTN(nil, []int{4, 0}) })
}

// TestShortSlicesPanicWithAMessage: a slice shorter than the plan used to fail
// deep inside a transform with a runtime index error; it now fails first, with
// the package's message, before any (bounds-trusting) kernel runs.
func TestShortSlicesPanicWithAMessage(t *testing.T) {
	const want = "fft: "
	p := NewPlan(64)
	mustPanicWith(t, "Plan.FFT short dst", want, func() { p.FFT(make([]complex128, 63), make([]complex128, 64)) })
	mustPanicWith(t, "Plan.FFT short src", want, func() { p.FFT(make([]complex128, 64), make([]complex128, 63)) })
	mustPanicWith(t, "Plan.IFFT short dst", want, func() { p.IFFT(make([]complex128, 1), make([]complex128, 64)) })
	rp := NewRealPlan(64)
	mustPanicWith(t, "RealPlan.RFFT short dst", want, func() { rp.RFFT(make([]complex128, 32), make([]float64, 64)) })
	mustPanicWith(t, "RealPlan.RFFT short src", want, func() { rp.RFFT(make([]complex128, 33), make([]float64, 63)) })
	mustPanicWith(t, "RealPlan.IRFFT short dst", want, func() { rp.IRFFT(make([]float64, 63), make([]complex128, 33)) })
}

// TestLongerSlicesStillWork: the length checks reject only what used to crash;
// a longer slice is still accepted, and IFFT no longer scales the values past
// the plan's length (it used to multiply all of dst by 1/n).
func TestLongerSlicesStillWork(t *testing.T) {
	const n = 16
	x := cmplxSignal(n)
	p := NewPlan(n)
	dst := make([]complex128, n+3)
	dst[n], dst[n+1], dst[n+2] = 7, 8, 9
	p.FFT(dst, x)
	p.IFFT(dst, dst[:n])
	for i := 0; i < n; i++ {
		if cmplx.Abs(dst[i]-x[i]) > 1e-12 {
			t.Fatalf("round trip index %d: %v want %v", i, dst[i], x[i])
		}
	}
	if dst[n] != 7 || dst[n+1] != 8 || dst[n+2] != 9 {
		t.Fatalf("IFFT touched values past the plan's length: %v", dst[n:])
	}
}

// FuzzPublicAPI drives the 1-D, 2-D and DCT/DST entry points with fuzzed
// lengths and shapes. The only panics allowed are the package's own ("fft: ..."), and every
// transform must round-trip.
func FuzzPublicAPI(f *testing.F) {
	for _, s := range [][3]int{{1, 1, 0}, {7, 3, 1}, {64, 16, 2}, {97, 5, 3}, {1000, 0, 4}, {0, 9, 5}, {12, 5, 8}, {9, 4, 10}, {2, 3, 13}, {5, 5, 19}} {
		f.Add(s[0], s[1], uint8(s[2]))
	}
	f.Fuzz(func(t *testing.T, a, b int, seed uint8) {
		// Bound the work, not the validation: oversized or negative values must
		// still be rejected cleanly, so they are passed through to shape checks.
		defer func() {
			if r := recover(); r != nil {
				msg, _ := r.(string)
				if !strings.HasPrefix(msg, "fft: ") {
					t.Fatalf("a=%d b=%d: unexpected panic %v", a, b, r)
				}
			}
		}()
		if a > 4096 || b > 4096 || a*b > 1<<16 && a > 0 && b > 0 {
			// Too large to run: only the shape validation is exercised.
			if a <= 0 || b <= 0 {
				FFTN(nil, []int{a, b})
			}
			return
		}
		n := a
		if n >= 0 {
			x := make([]complex128, n)
			for i := range x {
				x[i] = complex(float64((i*7+int(seed))%13)-6, float64((i*3+int(seed))%11)-5)
			}
			back := IFFT(FFT(x))
			for i := range x {
				if cmplx.Abs(back[i]-x[i]) > 1e-9*math.Max(1, float64(n)) {
					t.Fatalf("n=%d: FFT round trip index %d: %v want %v", n, i, back[i], x[i])
				}
			}
			r := make([]float64, n)
			for i := range r {
				r[i] = real(x[i])
			}
			rb := IRFFT(RFFT(r), n)
			for i := range r {
				if math.Abs(rb[i]-r[i]) > 1e-9*math.Max(1, float64(n)) {
					t.Fatalf("n=%d: RFFT round trip index %d: %v want %v", n, i, rb[i], r[i])
				}
			}
			// Single precision: the same signals (exact in float32), held to
			// the round-trip form of the bound the float32 tests use.
			x32 := make([]complex64, n)
			r32 := make([]float32, n)
			w64 := make([]float64, n)
			for i, v := range x {
				x32[i] = complex64(v)
				r32[i] = float32(real(v))
				w64[i] = real(v)
			}
			if e := relErr(IFFT32(FFT32(x32)), x); e > 4*bound32(n) {
				t.Fatalf("n=%d: FFT32 round trip error %g", n, e)
			}
			if e := relErrReal(IRFFT32(RFFT32(r32), n), w64); e > 4*bound32(n) {
				t.Fatalf("n=%d: RFFT32 round trip error %g", n, e)
			}
		}
		shape := []int{a, b}
		total := 0
		if a > 0 && b > 0 {
			total = a * b
		}
		d := make([]complex128, total)
		for i := range d {
			d[i] = complex(float64(i%5), float64(i%3))
		}
		bd := IFFTN(FFTN(d, shape), shape)
		for i := range d {
			if cmplx.Abs(bd[i]-d[i]) > 1e-9*math.Max(1, float64(total)) {
				t.Fatalf("shape %v: FFTN round trip index %d", shape, i)
			}
		}
		// DCT/DST of every type and norm, including the invalid ones (type 0
		// and 5, Norm(3)) and the lengths below a type's minimum: those must
		// panic with the package's message, the others round-trip.
		typ, norm := int(seed)%6, Norm(int(seed/6)%4)
		if a >= 0 {
			r := make([]float64, a)
			for i := range r {
				r[i] = float64((i*7+int(seed))%13) - 6
			}
			for _, cosine := range []bool{true, false} {
				fwd, inv := DST, IDST
				if cosine {
					fwd, inv = DCT, IDCT
				}
				back := inv(fwd(r, typ, norm), typ, norm)
				for i := range r {
					if math.Abs(back[i]-r[i]) > 1e-9*math.Max(1, float64(a)) {
						t.Fatalf("n=%d type %d %v cosine=%v: round trip index %d: %v want %v", a, typ, norm, cosine, i, back[i], r[i])
					}
				}
			}
			if total > 0 {
				dn := make([]float64, total)
				for i := range dn {
					dn[i] = float64(i%7) - 3
				}
				bn := IDCTN(DCTN(dn, shape, typ, norm), shape, typ, norm)
				for i := range dn {
					if math.Abs(bn[i]-dn[i]) > 1e-9*math.Max(1, float64(total)) {
						t.Fatalf("shape %v type %d %v: DCTN round trip index %d", shape, typ, norm, i)
					}
				}
			}
		}
		fuzzOptions(t, a, b, seed)
	})
}

// fuzzOptions drives the numpy-style entry points (Options, Hermitian, real
// N-D, shifts, NextFastLen) with fuzzed lengths, N, Norm and axes. Each call
// may panic only with the package's own "fft: ..." message, and each valid
// pair must round-trip. a and b are already bounded by FuzzPublicAPI.
func fuzzOptions(t *testing.T, a, b int, seed uint8) {
	norm := Norm(int(seed % 4)) // 3 is not a Norm: it must be refused cleanly
	axesChoices := [][]int{nil, {0}, {-1}, {1, 0}, {}, {2}, {0, -2}}
	axes := axesChoices[int(seed/4)%len(axesChoices)]
	try := func(name string, f func()) {
		t.Helper()
		defer func() {
			if r := recover(); r != nil {
				if msg, _ := r.(string); !strings.HasPrefix(msg, "fft: ") {
					t.Fatalf("%s a=%d b=%d seed=%d: unexpected panic %v", name, a, b, seed, r)
				}
			}
		}()
		f()
	}
	tol := func(n int) float64 { return 1e-9 * math.Max(1, float64(n)) }
	if a >= 0 {
		x := make([]complex128, a)
		r := make([]float64, a)
		for i := range x {
			x[i] = complex(float64((i*5+int(seed))%9)-4, float64((i*7+int(seed))%5)-2)
			r[i] = real(x[i])
		}
		o := Options{N: b, Norm: norm}
		try("FFTWith/IFFTWith", func() {
			X := FFTWith(x, o)
			back := IFFTWith(X, Options{N: len(X), Norm: norm})
			for i, v := range back {
				want := complex128(0)
				if i < len(x) {
					want = x[i]
				}
				if cmplx.Abs(v-want) > tol(len(X)) {
					t.Fatalf("FFTWith round trip a=%d N=%d %v index %d: %v want %v", a, b, norm, i, v, want)
				}
			}
		})
		try("RFFTWith/IRFFTWith", func() {
			n := o.N
			if n == 0 {
				n = a
			}
			back := IRFFTWith(RFFTWith(r, o), Options{N: n, Norm: norm})
			for i, v := range back {
				want := 0.0
				if i < len(r) {
					want = r[i]
				}
				if math.Abs(v-want) > tol(n) {
					t.Fatalf("RFFTWith round trip a=%d N=%d %v index %d: %v want %v", a, b, norm, i, v, want)
				}
			}
		})
		try("IHFFTWith/HFFTWith", func() {
			n := o.N
			if n == 0 {
				n = a
			}
			back := HFFTWith(IHFFTWith(r, o), Options{N: n, Norm: norm})
			for i, v := range back {
				want := 0.0
				if i < len(r) {
					want = r[i]
				}
				if math.Abs(v-want) > tol(n) {
					t.Fatalf("IHFFTWith round trip a=%d N=%d %v index %d: %v want %v", a, b, norm, i, v, want)
				}
			}
		})
		try("HFFT", func() { HFFT(x, b) })
		try("NextFastLen", func() {
			for _, real := range []bool{false, true} {
				m := NextFastLen(a, real)
				if m < a || (a > 1 && real && m%2 != 0) {
					t.Fatalf("NextFastLen(%d, %v) = %d", a, real, m)
				}
			}
		})
	}
	shape := []int{a, b}
	total := 0
	if a > 0 && b > 0 {
		total = a * b
	}
	d := make([]complex128, total)
	rd := make([]float64, total)
	for i := range d {
		d[i] = complex(float64(i%7)-3, float64(i%4))
		rd[i] = float64(i%6) - 2
	}
	o := Options{Axes: axes, Norm: norm}
	try("FFTNWith/IFFTNWith", func() {
		back := IFFTNWith(FFTNWith(d, shape, o), shape, o)
		for i := range d {
			if cmplx.Abs(back[i]-d[i]) > tol(total) {
				t.Fatalf("FFTNWith round trip shape %v axes %v index %d", shape, axes, i)
			}
		}
	})
	try("RFFTNWith/IRFFTNWith", func() {
		back := IRFFTNWith(RFFTNWith(rd, shape, o), shape, o)
		for i := range rd {
			if math.Abs(back[i]-rd[i]) > tol(total) {
				t.Fatalf("RFFTNWith round trip shape %v axes %v index %d", shape, axes, i)
			}
		}
	})
	try("FFTShiftN/IFFTShiftN", func() {
		back := IFFTShiftN(FFTShiftN(rd, shape, axes), shape, axes)
		for i := range rd {
			if back[i] != rd[i] {
				t.Fatalf("FFTShiftN round trip shape %v axes %v index %d", shape, axes, i)
			}
		}
	})
}
