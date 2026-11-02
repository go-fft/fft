package fft

import (
	"math"
	"math/cmplx"
	"strings"
	"sync"
	"testing"
)

// chaosSizes mixes every route: powers of two, mixed radix, Rader primes,
// Bluestein-sized primes, and a few tiny lengths.
var chaosSizes = []int{1, 2, 3, 5, 7, 8, 12, 16, 60, 97, 128, 360, 1009, 1024, 2310, 4093}

// TestChaosSharedPlans: one Plan, RealPlan and RealPlan2 per size, shared by
// many goroutines at once (run under -race), FMA and default, each goroutine
// with its own buffers; every result must equal the serial one bit for bit.
func TestChaosSharedPlans(t *testing.T) {
	for _, fma := range []bool{false, true} {
		o := Options{FMA: fma}
		for _, n := range chaosSizes {
			p := NewPlanWith(n, o)
			rp := NewRealPlanWith(n, o)
			x := make([]complex128, n)
			r := make([]float64, n)
			for i := range x {
				x[i] = complex(math.Sin(float64(i)*0.37), math.Cos(float64(i)*0.11))
				r[i] = real(x[i])
			}
			wantF := p.FFT(make([]complex128, n), x)
			wantI := p.IFFT(make([]complex128, n), x)
			wantR := rp.RFFT(make([]complex128, n/2+1), r)
			wantB := rp.IRFFT(make([]float64, n), wantR)
			var wg sync.WaitGroup
			for g := 0; g < 24; g++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for it := 0; it < 20; it++ {
						gotF := p.FFT(make([]complex128, n), x)
						gotI := p.IFFT(make([]complex128, n), x)
						gotR := rp.RFFT(make([]complex128, n/2+1), r)
						gotB := rp.IRFFT(make([]float64, n), gotR)
						// in place, too
						ip := append([]complex128(nil), x...)
						p.FFT(ip, ip)
						for i := range wantF {
							if gotF[i] != wantF[i] || gotI[i] != wantI[i] || ip[i] != wantF[i] {
								t.Errorf("fma=%v n=%d: complex result differs at %d", fma, n, i)
								return
							}
						}
						for i := range wantR {
							if gotR[i] != wantR[i] {
								t.Errorf("fma=%v n=%d: RFFT differs at %d", fma, n, i)
								return
							}
						}
						for i := range wantB {
							if gotB[i] != wantB[i] {
								t.Errorf("fma=%v n=%d: IRFFT differs at %d", fma, n, i)
								return
							}
						}
					}
				}()
			}
			wg.Wait()
		}
		rp2 := NewRealPlan2With(24, 30, o)
		src := make([]float64, 24*30)
		for i := range src {
			src[i] = math.Sin(float64(i))
		}
		want := rp2.RFFT(make([]complex128, rp2.SpectrumLen()), src)
		var wg sync.WaitGroup
		for g := 0; g < 16; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for it := 0; it < 10; it++ {
					got := rp2.RFFT(make([]complex128, rp2.SpectrumLen()), src)
					for i := range want {
						if got[i] != want[i] {
							t.Errorf("fma=%v RealPlan2 differs at %d", fma, i)
							return
						}
					}
				}
			}()
		}
		wg.Wait()
	}
}

// TestChaosNonFinite: NaN and Inf inputs must neither panic nor hang, and a
// plan that has seen them must still give the exact answer afterwards (no
// state left in scratch pools).
func TestChaosNonFinite(t *testing.T) {
	bad := []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.MaxFloat64, math.SmallestNonzeroFloat64, -0.0}
	for _, n := range chaosSizes {
		p := NewPlan(n)
		rp := NewRealPlan(n)
		good := make([]complex128, n)
		for i := range good {
			good[i] = complex(float64(i%5), float64(i%3))
		}
		gr := make([]float64, n)
		for i := range gr {
			gr[i] = float64(i % 7)
		}
		wantC := p.FFT(make([]complex128, n), good)
		wantR := rp.RFFT(make([]complex128, n/2+1), gr)
		for _, b := range bad {
			for _, pos := range []int{0, n / 2, n - 1} {
				x := append([]complex128(nil), good...)
				x[pos] = complex(b, b)
				p.FFT(make([]complex128, n), x)
				p.IFFT(x, x)
				r := append([]float64(nil), gr...)
				r[pos] = b
				s := rp.RFFT(make([]complex128, n/2+1), r)
				rp.IRFFT(make([]float64, n), s)
				FFT32(complex64Of(x))
				// a NaN/Inf spectrum into the inverse real transform
				sp := make([]complex128, n/2+1)
				sp[pos/2] = complex(b, -b)
				rp.IRFFT(make([]float64, n), sp)
			}
		}
		gotC := p.FFT(make([]complex128, n), good)
		gotR := rp.RFFT(make([]complex128, n/2+1), gr)
		for i := range wantC {
			if gotC[i] != wantC[i] {
				t.Fatalf("n=%d: plan state poisoned by non-finite input at %d", n, i)
			}
		}
		for i := range wantR {
			if gotR[i] != wantR[i] {
				t.Fatalf("n=%d: real plan state poisoned at %d", n, i)
			}
		}
	}
}

func complex64Of(x []complex128) []complex64 {
	y := make([]complex64, len(x))
	for i, v := range x {
		y[i] = complex64(v)
	}
	return y
}

// TestChaosLengths: zero, negative and absurd lengths either work or panic
// with the package's own message; they never allocate their way into a fatal
// error or run an index out of range.
func TestChaosLengths(t *testing.T) {
	lens := []int{-1 << 62, -1 << 31, -2, -1, 0, 1, 1 << 62, math.MaxInt}
	try := func(name string, f func()) {
		if testing.Verbose() {
			println("chaos:", name)
		}
		defer func() {
			if r := recover(); r != nil {
				// An absurd length reaches make and fails with the runtime's
				// recoverable "makeslice: len out of range" (an ordinary
				// panic, not the fatal out-of-memory); anything else must be
				// the package's own message.
				msg, _ := r.(string)
				if e, ok := r.(error); ok {
					msg = e.Error()
					if strings.Contains(msg, "makeslice: len out of range") {
						return
					}
				}
				if !strings.HasPrefix(msg, "fft: ") {
					t.Errorf("%s: unexpected panic %v", name, r)
				}
			}
		}()
		f()
	}
	for _, n := range lens {
		if testing.Verbose() {
			println("chaos n =", n)
		}
		try("NewPlan", func() { NewPlan(n) })
		try("NewPlan32", func() { NewPlan32(n) })
		try("NewRealPlan", func() { NewRealPlan(n) })
		try("NewRealPlan32", func() { NewRealPlan32(n) })
		try("NewRealPlan2", func() { NewRealPlan2(n, 2) })
		try("NewRealPlan2_32", func() { NewRealPlan2_32(2, n) })
		try("NewPlanN", func() { NewPlanN(n) })
		try("NewPlanN32", func() { NewPlanN32(2, n) })
		try("FFTN", func() { FFTN(make([]complex128, 4), []int{n}) })
		try("RFFT", func() { IRFFT(make([]complex128, 4), n) })
		try("DCT", func() { DCT(make([]float64, 4), 2, NormBackward) })
		try("NextFastLen", func() { NextFastLen(n, true) })
		try("FFTFreq", func() { FFTFreq(n, 1) })
		try("FFTShiftN", func() { FFTShiftN(make([]float64, 3), []int{n}, nil) })
	}
}

// TestChaosPartialOverlap: dst and src that overlap without being the same
// slice are not part of the contract, but they must not crash, and the
// identical-slice case must stay exact.
func TestChaosPartialOverlap(t *testing.T) {
	for _, n := range chaosSizes {
		p := NewPlan(n)
		buf := make([]complex128, 2*n+2)
		for _, off := range []int{1, n / 2, n - 1, n} {
			if off < 0 || off+n > len(buf) {
				continue
			}
			for i := range buf {
				buf[i] = complex(float64(i%9), 1)
			}
			p.FFT(buf[off:off+n], buf[:n])
			for i := range buf {
				buf[i] = complex(float64(i%9), 1)
			}
			p.IFFT(buf[:n], buf[off:off+n])
		}
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(float64(i%9), 1)
		}
		want := p.FFT(make([]complex128, n), x)
		p.FFT(x, x)
		for i := range x {
			if cmplx.Abs(x[i]-want[i]) != 0 {
				t.Fatalf("n=%d: in-place differs at %d", n, i)
			}
		}
	}
}
