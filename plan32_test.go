package fft

import (
	"math"
	"math/rand/v2"
	"sync"
	"testing"
	"unsafe"
)

// eps32 is FLT_EPSILON, 2^-23.
var eps32 = math.Ldexp(1, -23)

// bound32 is the accuracy every single-precision transform is held to: the
// 2-norm of the error, relative to the 2-norm of the exact result, at most
// eps32·log2(n) (eps32 for n <= 2). The worst measured on the sizes below is
// 0.49 of it (n = 2, NormOrtho, whose 1/sqrt(2) is itself rounded), and 0.14
// over the lengths of 1024 points and more; a radix-2 FFT with exact twiddles has an RMS
// relative error of about eps·sqrt(log2 n) and a worst case of O(eps·log2 n)
// (Higham, Accuracy and Stability of Numerical Algorithms, §24.1).
func bound32(n int) float64 { return eps32 * math.Max(1, math.Log2(float64(n))) }

// relErr returns ||got - want||_2 / ||want||_2 (0 when both are empty or
// want is zero and got is too).
func relErr(got []complex64, want []complex128) float64 {
	var num, den float64
	for i, w := range want {
		d := complex128(got[i]) - w
		num += real(d)*real(d) + imag(d)*imag(d)
		den += real(w)*real(w) + imag(w)*imag(w)
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}

// relErrReal is relErr for real vectors.
func relErrReal(got []float32, want []float64) float64 {
	var num, den float64
	for i, w := range want {
		d := float64(got[i]) - w
		num += d * d
		den += w * w
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}

// signal32 returns n random complex64 samples in [-1,1)² and the same values
// widened to complex128 (exactly: widening is exact).
func signal32(n int, seed uint64) ([]complex64, []complex128) {
	r := rand.New(rand.NewPCG(seed, uint64(n)))
	x := make([]complex64, n)
	w := make([]complex128, n)
	for i := range x {
		x[i] = complex(float32(r.Float64()*2-1), float32(r.Float64()*2-1))
		w[i] = complex128(x[i])
	}
	return x, w
}

// scaled returns s·x.
func scaled(x []complex128, s float64) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = v * complex(s, 0)
	}
	return out
}

// naiveIDFT is the unnormalized inverse DFT, by definition.
func naiveIDFT(x []complex128) []complex128 {
	c := make([]complex128, len(x))
	for i, v := range x {
		c[i] = complexConj(v)
	}
	out := naiveDFT(c)
	for i, v := range out {
		out[i] = complexConj(v)
	}
	return out
}

// worst32 records the largest error seen, as a fraction of bound32, overall
// and over the lengths of at least 1024 points.
type worst32 struct {
	all, large float64
	n          int
	what       string
}

func (w *worst32) note(e float64, n int, m Norm, what string) {
	if e > w.all {
		w.all, w.n, w.what = e, n, m.String()+" "+what
	}
	if n >= 1024 {
		w.large = math.Max(w.large, e)
	}
}

func (w *worst32) log(t *testing.T) {
	t.Helper()
	t.Logf("worst error %.3f of eps32·log2(n) (n=%d, %s); %.3f for n >= 1024", w.all, w.n, w.what, w.large)
}

var allNorms = []Norm{NormBackward, NormOrtho, NormForward}

// sizes32 covers every length 0..130 (so every radix pass, with and without
// twiddles, the general pass for 11 and 13, Rader for 17, 97, 101, …, and
// Bluestein for 23, 46, 47, …) plus larger smooth, prime and composite ones.
func sizes32(t *testing.T) []int {
	var s []int
	for n := 0; n <= 130; n++ {
		s = append(s, n)
	}
	s = append(s, 256, 641, 1000, 1009, 1024, 1080, 1920, 2017, 4096, 10007, 12289, 65521, 65536, 240240)
	if !testing.Short() {
		s = append(s, 100003, 1<<18, 1<<20)
	}
	return s
}

// TestPlan32AgainstTheDefinition holds the forward and inverse transforms,
// under every Norm, to bound32 against the DFT computed in float64: the
// naive O(n²) sum up to 300 points, the float64 transform above.
func TestPlan32AgainstTheDefinition(t *testing.T) {
	var w worst32
	for _, n := range sizes32(t) {
		x, x64 := signal32(n, 1)
		var fwd, inv []complex128
		if n <= 300 {
			fwd, inv = naiveDFT(x64), naiveIDFT(x64)
		} else {
			fwd = FFT(x64)
			inv = scaled(IFFT(x64), float64(n))
		}
		p := NewPlan32(n)
		if p.Len() != n {
			t.Fatalf("Len() = %d, want %d", p.Len(), n)
		}
		for _, m := range allNorms {
			dst := make([]complex64, n)
			got := p.FFTNorm(dst, x, m)
			if e := relErr(got, scaled(fwd, m.scale(n, false))) / bound32(n); e > 1 {
				t.Errorf("n=%d %v forward: error %.3g× the bound", n, m, e)
			} else {
				w.note(e, n, m, "forward")
			}
			got = p.IFFTNorm(dst, x, m)
			if e := relErr(got, scaled(inv, m.scale(n, true))) / bound32(n); e > 1 {
				t.Errorf("n=%d %v inverse: error %.3g× the bound", n, m, e)
			} else {
				w.note(e, n, m, "inverse")
			}
		}
		// The default-normalized entry points are NormBackward.
		if e := relErr(FFT32(x), fwd); e > bound32(n) {
			t.Errorf("n=%d FFT32: error %g", n, e)
		}
		if e := relErr(IFFT32(x), scaled(inv, NormBackward.scale(n, true))); e > bound32(n) {
			t.Errorf("n=%d IFFT32: error %g", n, e)
		}
	}
	w.log(t)
}

// TestRealPlan32AgainstTheDefinition holds RFFT and IRFFT, under every Norm,
// to bound32 against the float64 real transforms (themselves checked against
// the definition by rfft_test.go) and, up to 300 points, the naive DFT.
func TestRealPlan32AgainstTheDefinition(t *testing.T) {
	var w worst32
	for _, n := range sizes32(t) {
		x, x64 := signal32(n, 2)
		re := make([]float32, n)
		re64 := make([]float64, n)
		for i := range x {
			re[i], re64[i] = real(x[i]), real(x64[i])
		}
		h := n/2 + 1
		var fwd []complex128
		if n <= 300 {
			c := make([]complex128, n)
			for i, v := range re64 {
				c[i] = complex(v, 0)
			}
			fwd = naiveDFT(c)
		} else {
			fwd = RFFT(re64)
		}
		if n > 0 {
			fwd = fwd[:h]
		}
		// A Hermitian half spectrum to invert: the forward one of the signal.
		spec := make([]complex64, h)
		spec64 := make([]complex128, h)
		for i := 0; i < h && n > 0; i++ {
			spec[i] = x[i]
			spec64[i] = x64[i]
		}
		inv64 := IRFFT(spec64, n) // normalized by n
		p := NewRealPlan32(n)
		if p.Len() != n {
			t.Fatalf("Len() = %d, want %d", p.Len(), n)
		}
		for _, m := range allNorms {
			got := p.RFFTNorm(make([]complex64, h), re, m)
			if len(got) != len(fwd) {
				t.Fatalf("n=%d RFFT: %d bins, want %d", n, len(got), len(fwd))
			}
			if e := relErr(got, scaled(fwd, m.scale(n, false))) / bound32(n); e > 1 {
				t.Errorf("n=%d %v RFFT: error %.3g× the bound", n, m, e)
			} else {
				w.note(e, n, m, "RFFT")
			}
			want := make([]float64, n)
			for i, v := range inv64 {
				want[i] = v * float64(n) * m.scale(n, true)
			}
			back := p.IRFFTNorm(make([]float32, n), spec, m)
			if e := relErrReal(back, want) / bound32(n); e > 1 {
				t.Errorf("n=%d %v IRFFT: error %.3g× the bound", n, m, e)
			} else {
				w.note(e, n, m, "IRFFT")
			}
		}
		if n > 0 {
			if e := relErr(RFFT32(re), fwd); e > bound32(n) {
				t.Errorf("n=%d RFFT32: error %g", n, e)
			}
			if e := relErrReal(IRFFT32(spec, n), inv64); e > bound32(n) {
				t.Errorf("n=%d IRFFT32: error %g", n, e)
			}
		}
	}
	w.log(t)
}

// TestPlan32RoundTrip: under every Norm, the inverse of the forward transform
// returns the input, in place (dst aliasing src).
func TestPlan32RoundTrip(t *testing.T) {
	for _, n := range []int{1, 2, 7, 12, 16, 23, 97, 128, 1000, 4096} {
		x, x64 := signal32(n, 3)
		p := NewPlan32(n)
		for _, m := range allNorms {
			y := append([]complex64(nil), x...)
			p.FFTNorm(y, y, m)
			p.IFFTNorm(y, y, m)
			if e := relErr(y, x64); e > 2*bound32(n) {
				t.Errorf("n=%d %v: round trip error %g", n, m, e)
			}
			r := make([]float32, n)
			for i := range r {
				r[i] = real(x[i])
			}
			rp := NewRealPlan32(n)
			back := rp.IRFFTNorm(make([]float32, n), rp.RFFTNorm(make([]complex64, n/2+1), r, m), m)
			want := make([]float64, n)
			for i := range want {
				want[i] = real(x64[i])
			}
			if e := relErrReal(back, want); e > 2*bound32(n) {
				t.Errorf("n=%d %v: real round trip error %g", n, m, e)
			}
		}
	}
}

// TestPlan32SmallExact: transforms whose arithmetic is exact in float32 come
// out exact.
func TestPlan32SmallExact(t *testing.T) {
	got := FFT32([]complex64{1, 2, 3, 4})
	want := []complex64{10, -2 + 2i, -2, -2 - 2i}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("FFT32([1 2 3 4]) = %v, want %v", got, want)
		}
	}
	r := RFFT32([]float32{1, 2, 3, 4})
	for i, w := range want[:3] {
		if r[i] != w {
			t.Fatalf("RFFT32([1 2 3 4]) = %v, want %v", r, want[:3])
		}
	}
	if b := IRFFT32(r, 4); b[0] != 1 || b[1] != 2 || b[2] != 3 || b[3] != 4 {
		t.Fatalf("IRFFT32 = %v, want [1 2 3 4]", b)
	}
	if one := FFT32([]complex64{3 - 1i}); one[0] != 3-1i {
		t.Fatalf("FFT32 of one point = %v", one)
	}
	if one := RFFT32([]float32{5}); len(one) != 1 || one[0] != 5 {
		t.Fatalf("RFFT32 of one point = %v", one)
	}
	if one := IRFFT32([]complex64{5 + 7i}, 1); one[0] != 5 {
		t.Fatalf("IRFFT32 of one bin = %v", one)
	}
}

// TestPlan32Empty: empty inputs give empty, non-nil results.
func TestPlan32Empty(t *testing.T) {
	if r := FFT32(nil); r == nil || len(r) != 0 {
		t.Errorf("FFT32(nil) = %#v", r)
	}
	if r := IFFT32([]complex64{}); r == nil || len(r) != 0 {
		t.Errorf("IFFT32(empty) = %#v", r)
	}
	if r := RFFT32(nil); r == nil || len(r) != 0 {
		t.Errorf("RFFT32(nil) = %#v", r)
	}
	for _, n := range []int{0, -3} {
		if r := IRFFT32([]complex64{1}, n); r == nil || len(r) != 0 {
			t.Errorf("IRFFT32(_, %d) = %#v", n, r)
		}
	}
	rp := NewRealPlan32(0)
	if r := rp.RFFT(nil, nil); len(r) != 0 {
		t.Errorf("RealPlan32(0).RFFT = %v", r)
	}
	if r := rp.IRFFT(nil, nil); len(r) != 0 {
		t.Errorf("RealPlan32(0).IRFFT = %v", r)
	}
}

// TestPlan32Panics: short slices and an unknown Norm panic with the package's
// message.
func TestPlan32Panics(t *testing.T) {
	p := NewPlan32(16)
	mustPanicWith(t, "Plan32.FFT short dst", "fft: ", func() { p.FFT(make([]complex64, 15), make([]complex64, 16)) })
	mustPanicWith(t, "Plan32.IFFT short src", "fft: ", func() { p.IFFT(make([]complex64, 16), make([]complex64, 15)) })
	mustPanicWith(t, "Plan32.FFTNorm bad norm", "fft: unknown Norm", func() { p.FFTNorm(make([]complex64, 16), make([]complex64, 16), Norm(9)) })
	rp := NewRealPlan32(16)
	mustPanicWith(t, "RealPlan32.RFFT short dst", "fft: ", func() { rp.RFFT(make([]complex64, 8), make([]float32, 16)) })
	mustPanicWith(t, "RealPlan32.RFFT short src", "fft: ", func() { rp.RFFT(make([]complex64, 9), make([]float32, 15)) })
	mustPanicWith(t, "RealPlan32.IRFFT short dst", "fft: ", func() { rp.IRFFT(make([]float32, 15), make([]complex64, 9)) })
	mustPanicWith(t, "RealPlan32.IRFFTNorm bad norm", "fft: unknown Norm", func() { rp.IRFFTNorm(make([]float32, 16), nil, -1) })
}

// TestPlan32LongerSlices: longer slices are accepted and nothing past the
// plan's length is touched.
func TestPlan32LongerSlices(t *testing.T) {
	const n = 12
	x, x64 := signal32(n, 4)
	dst := make([]complex64, n+2)
	dst[n], dst[n+1] = 7, 8
	NewPlan32(n).IFFT(dst, append(x, 1, 2))
	if dst[n] != 7 || dst[n+1] != 8 {
		t.Fatalf("IFFT touched values past the plan's length: %v", dst[n:])
	}
	if e := relErr(dst[:n], scaled(naiveIDFT(x64), 1.0/n)); e > bound32(n) {
		t.Fatalf("IFFT error %g", e)
	}
}

// TestIRFFT32ShortSpectrum: missing bins count as zero, and bins past N/2
// are ignored, as for IRFFT.
func TestIRFFT32ShortSpectrum(t *testing.T) {
	for _, n := range []int{8, 9, 10} {
		spec := []complex64{4, 1 - 2i}
		spec64 := []complex128{4, 1 - 2i}
		if e := relErrReal(IRFFT32(spec, n), IRFFT(spec64, n)); e > bound32(n) {
			t.Errorf("n=%d short spectrum: error %g", n, e)
		}
		long := make([]complex64, n)
		long64 := make([]complex128, n)
		for i := range long {
			long[i] = complex(float32(i), 1)
			long64[i] = complex128(long[i])
		}
		if e := relErrReal(IRFFT32(long, n), IRFFT(long64, n)); e > bound32(n) {
			t.Errorf("n=%d long spectrum: error %g", n, e)
		}
	}
}

// TestPlan32NoMutation: the package-level functions leave their input alone.
func TestPlan32NoMutation(t *testing.T) {
	x, _ := signal32(30, 5)
	keep := append([]complex64(nil), x...)
	FFT32(x)
	IFFT32(x)
	r := []float32{1, 2, 3, 4, 5, 6}
	RFFT32(r)
	s := []complex64{1, 2, 3, 4}
	IRFFT32(s, 6)
	for i := range x {
		if x[i] != keep[i] {
			t.Fatal("FFT32/IFFT32 modified the input")
		}
	}
	if r[0] != 1 || r[5] != 6 || s[0] != 1 || s[3] != 4 {
		t.Fatal("RFFT32/IRFFT32 modified the input")
	}
}

// TestPlan32Cache: the package-level functions reuse one plan per length.
func TestPlan32Cache(t *testing.T) {
	if cachedPlan32(48) != cachedPlan32(48) {
		t.Error("cachedPlan32 built two plans for one length")
	}
	if cachedRealPlan32(48) != cachedRealPlan32(48) {
		t.Error("cachedRealPlan32 built two plans for one length")
	}
}

// TestPlan32Concurrent: one plan used from many goroutines gives every caller
// the same answer (run with -race).
func TestPlan32Concurrent(t *testing.T) {
	for _, n := range []int{1024, 1009, 23} {
		x, _ := signal32(n, 6)
		p := NewPlan32(n)
		want := p.FFT(make([]complex64, n), x)
		var wg sync.WaitGroup
		for g := 0; g < 8; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				got := p.FFT(make([]complex64, n), x)
				for i := range got {
					if got[i] != want[i] {
						t.Errorf("n=%d: concurrent result differs at %d", n, i)
						return
					}
				}
			}()
		}
		wg.Wait()
	}
}

// TestPlan32Engines: lengths land on the engine their factorization calls for.
func TestPlan32Engines(t *testing.T) {
	for _, c := range []struct {
		n                    int
		sk, rader, bluestein bool
	}{{1, false, false, false}, {1080, true, false, false}, {1009, false, true, false}, {23, false, false, true}, {2 * 23, false, false, true}} {
		p := NewPlan32(c.n)
		if (p.sk != nil) != c.sk || (p.rader != nil) != c.rader || (p.bluestein != nil) != c.bluestein {
			t.Errorf("n=%d: engines sk=%v rader=%v bluestein=%v", c.n, p.sk != nil, p.rader != nil, p.bluestein != nil)
		}
	}
}

// TestAsComplex64Layout pins what asComplex64 relies on: a complex64 is its
// real then its imaginary float32.
func TestAsComplex64Layout(t *testing.T) {
	f := []float32{1, 2, 3, 4}
	c := asComplex64(f)
	if len(c) != 2 || c[0] != 1+2i || c[1] != 3+4i {
		t.Fatalf("asComplex64(%v) = %v", f, c)
	}
	if unsafe.Sizeof(complex64(0)) != 2*unsafe.Sizeof(float32(0)) {
		t.Fatal("complex64 is not two float32s")
	}
}
