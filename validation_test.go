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

// FuzzPublicAPI drives the 1-D and 2-D entry points with fuzzed lengths and
// shapes. The only panics allowed are the package's own ("fft: ..."), and every
// transform must round-trip.
func FuzzPublicAPI(f *testing.F) {
	for _, s := range [][3]int{{1, 1, 0}, {7, 3, 1}, {64, 16, 2}, {97, 5, 3}, {1000, 0, 4}, {0, 9, 5}} {
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
	})
}
