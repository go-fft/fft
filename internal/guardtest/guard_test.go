// Package guardtest holds the black-box tests of the length guards. It lives
// apart from the root package because the root package's own tests assume a
// 64-bit int (constants such as 1<<62) and do not compile on 32-bit targets,
// while these must run there: on a 32-bit target math.MaxInt is 2^31-1, so the
// bound math.MaxInt/16 is 2^27 and the guard has to fire for lengths a 64-bit
// host would still try to allocate.
package guardtest

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/go-fft/fft"
)

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

// TestHugeLengthsPanicWithAMessage: a length whose byte count overflows int is
// refused with the package's message before anything is allocated, by every
// public constructor and length-taking entry point that used to reach make
// and fail with the runtime's "makeslice: len out of range". Lengths below the
// bound are not capped (see SECURITY.md).
func TestHugeLengthsPanicWithAMessage(t *testing.T) {
	const huge = math.MaxInt
	type tc struct {
		name, want string
		f          func()
	}
	cases := []tc{
		{"NewPlan", "fft: NewPlan: length ", func() { fft.NewPlan(huge) }},
		{"NewPlan even", "fft: NewPlan: length ", func() { fft.NewPlan(huge - 1) }},
		{"NewPlanWith", "fft: NewPlan: length ", func() { fft.NewPlanWith(huge, fft.Options{}) }},
		{"NewPlanWith FMA", "fft: NewPlan: length ", func() { fft.NewPlanWith(huge, fft.Options{FMA: true}) }},
		{"NewPlan32", "fft: NewPlan32: length ", func() { fft.NewPlan32(huge) }},
		{"NewRealPlan", "fft: NewRealPlan: length ", func() { fft.NewRealPlan(huge) }},
		{"NewRealPlan even", "fft: NewRealPlan: length ", func() { fft.NewRealPlan(huge - 1) }},
		{"NewRealPlanWith", "fft: NewRealPlan: length ", func() { fft.NewRealPlanWith(huge, fft.Options{FMA: true}) }},
		{"NewRealPlan32", "fft: NewRealPlan32: length ", func() { fft.NewRealPlan32(huge) }},
		{"NewRealPlan2", "fft: NewRealPlan2: length ", func() { fft.NewRealPlan2(huge, 1) }},
		{"NewRealPlan2With", "fft: NewRealPlan2: length ", func() { fft.NewRealPlan2With(1, huge, fft.Options{}) }},
		{"NewRealPlan2_32", "fft: NewRealPlan2_32: length ", func() { fft.NewRealPlan2_32(huge, 1) }},
		{"NewPlanN", "fft: NewPlanN: length ", func() { fft.NewPlanN(huge) }},
		{"NewPlanN two axes", "fft: NewPlanN: length ", func() { fft.NewPlanN(huge/2, 2) }},
		{"NewPlanNWith", "fft: NewPlanN: length ", func() { fft.NewPlanNWith([]int{huge}, fft.Options{}) }},
		{"NewPlanN32", "fft: NewPlanN32: length ", func() { fft.NewPlanN32(huge) }},
		{"IRFFT", "fft: IRFFT: length ", func() { fft.IRFFT(nil, huge) }},
		{"IRFFTWith", "fft: IRFFT: length ", func() { fft.IRFFTWith(nil, fft.Options{N: huge}) }},
		{"HFFT", "fft: IRFFT: length ", func() { fft.HFFT(nil, huge) }},
		{"HFFTWith", "fft: IRFFT: length ", func() { fft.HFFTWith(nil, fft.Options{N: huge}) }},
		{"IRFFT32", "fft: NewRealPlan32: length ", func() { fft.IRFFT32(nil, huge) }},
		{"IRFFT32With", "fft: NewRealPlan32: length ", func() { fft.IRFFT32With(nil, fft.Options{N: huge}) }},
		{"HFFT32", "fft: NewRealPlan32: length ", func() { fft.HFFT32(nil, huge) }},
		{"FFTWith", "fft: FFTWith: length ", func() { fft.FFTWith(nil, fft.Options{N: huge}) }},
		{"IFFTWith", "fft: IFFTWith: length ", func() { fft.IFFTWith(nil, fft.Options{N: huge}) }},
		{"RFFTWith", "fft: RFFTWith: length ", func() { fft.RFFTWith(nil, fft.Options{N: huge}) }},
		{"FFT32With", "fft: FFT32With: length ", func() { fft.FFT32With(nil, fft.Options{N: huge}) }},
		{"IFFT32With", "fft: IFFT32With: length ", func() { fft.IFFT32With(nil, fft.Options{N: huge}) }},
		{"RFFT32With", "fft: RFFT32With: length ", func() { fft.RFFT32With(nil, fft.Options{N: huge}) }},
		{"FFTFreq", "fft: FFTFreq: length ", func() { fft.FFTFreq(huge, 1) }},
		{"RFFTFreq", "fft: RFFTFreq: length ", func() { fft.RFFTFreq(huge, 1) }},
		{"Hann", "fft: window: length ", func() { fft.Hann(huge) }},
		{"Hamming", "fft: window: length ", func() { fft.Hamming(huge) }},
		{"Blackman", "fft: window: length ", func() { fft.Blackman(huge) }},
		{"BlackmanHarris", "fft: window: length ", func() { fft.BlackmanHarris(huge) }},
		{"Bartlett", "fft: Bartlett: length ", func() { fft.Bartlett(huge) }},
	}
	cases = append(cases,
		tc{"NewMDCTPlan", "fft: ", func() { fft.NewMDCTPlan(huge - 1) }},
		tc{"NewMDCTPlan32", "fft: ", func() { fft.NewMDCTPlan32(huge - 1) }},
		tc{"NewMDCTPlan even m", "fft: ", func() { fft.NewMDCTPlan(huge - 3) }},
	)
	for typ := 1; typ <= 4; typ++ {
		typ := typ
		cases = append(cases,
			tc{"NewDCTPlan " + strconv.Itoa(typ), "fft: DCT-", func() { fft.NewDCTPlan(huge, typ) }},
			tc{"NewDSTPlan " + strconv.Itoa(typ), "fft: DST-", func() { fft.NewDSTPlan(huge, typ) }},
			tc{"NewDCTPlan32 " + strconv.Itoa(typ), "fft: DCT-", func() { fft.NewDCTPlan32(huge, typ) }},
			tc{"NewDSTPlan32 " + strconv.Itoa(typ), "fft: DST-", func() { fft.NewDSTPlan32(huge, typ) }},
		)
	}
	for _, c := range cases {
		mustPanicWith(t, c.name, c.want, c.f)
	}
}

// TestMDCTTwiddlesDoNotOverflow: for odd N/2 the MDCT's post-twiddle angle is
// built from (m+1)(2k+1), which exceeds 2^31 from m = 32,769 (N = 65,538). The
// IMDCT of a single coefficient is one cosine of the definition, so every output
// is checked against it with the argument reduced in int64. This is the one place
// the 32-bit lanes (386) exercise the MDCT.
func TestMDCTTwiddlesDoNotOverflow(t *testing.T) {
	const n = 65538
	const m = n / 2
	p := fft.NewMDCTPlan(n)
	out := make([]float64, n)
	for _, k := range []int{m - 1, 20000, 1} {
		src := make([]float64, m)
		src[k] = 1
		p.IMDCT(out, src, 1)
		for i, got := range out {
			r := (int64(2*i+1+m) * int64(2*k+1)) % int64(8*m)
			want := math.Cos(2 * math.Pi * float64(r) / float64(8*m))
			if math.Abs(got-want) > 1e-9 {
				t.Fatalf("k=%d: IMDCT[%d] = %g, want %g", k, i, got, want)
			}
		}
	}
}

// TestLengthsBelowTheBoundAreNotCapped: the guard is the overflow bound only;
// a length it admits builds as before.
func TestLengthsBelowTheBoundAreNotCapped(t *testing.T) {
	for _, n := range []int{1 << 16, 1<<16 + 1} {
		if fft.NewPlan(n).Len() != n || fft.NewRealPlan(n).Len() != n {
			t.Fatalf("n=%d: plan not built", n)
		}
	}
}
