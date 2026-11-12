package fft

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"strconv"
)

// The cases the exact-arithmetic tests share: every entry point that takes
// Options (float32 and the plan constructors included), over the lengths and
// shapes of every engine, each as the default call and as the same call with
// Options.Exact. exact(o) is o with Exact set; the file exact_test.go defines
// it (a copy of this file runs against v0.28.16, which has no Exact, where it
// returns o: see exact_golden_test.go).

// exactCase is one transform called with and without Options.Exact.
type exactCase struct {
	name       string
	def, exact func() any
}

func exactCases() []exactCase {
	var cs []exactCase
	key := ""
	add := func(name string, def, exact func() any) {
		cs = append(cs, exactCase{name + "/" + key, def, exact})
	}
	lengths := []int{1, 2, 17, 64, 360, 1000, 1001, 1009, 2018, 4096, 4099, 16384}
	shapes := [][]int{{12, 20}, {8, 8, 8}, {17, 9}, {30, 31}, {1009, 3}, {64, 64}, {16, 16, 16}}
	switch runtime.GOARCH {
	case "s390x", "ppc64le", "loong64", "riscv64":
		// Under qemu the whole set takes most of the CI timeout (no golden
		// digests exist for these architectures either).
		lengths, shapes = lengths[:5], shapes[:3]
	default:
		if raceEnabled {
			// -race with coverage on a slow runner: the suite is near the 10
			// minutes go test allows; the long lengths run in the other lanes.
			lengths, shapes = lengths[:9], shapes[:5]
		}
	}
	for _, n := range lengths {
		key = "n=" + strconv.Itoa(n)
		x := fmaSignal(n, uint64(n))
		r := fmaReal(n, uint64(n))
		h := RFFT(r)
		x32 := make([]complex64, n)
		r32 := make([]float32, n)
		for i := range x {
			x32[i] = complex64(x[i])
			r32[i] = float32(r[i])
		}
		h32 := RFFT32(r32)
		ex := exact(Options{})
		exN := exact(Options{N: n})
		add("FFT", func() any { return FFT(x) }, func() any { return FFTWith(x, ex) })
		add("IFFT", func() any { return IFFT(x) }, func() any { return IFFTWith(x, ex) })
		add("RFFT", func() any { return RFFT(r) }, func() any { return RFFTWith(r, ex) })
		add("IRFFT", func() any { return IRFFT(h, n) }, func() any { return IRFFTWith(h, exN) })
		add("HFFT", func() any { return HFFT(h, n) }, func() any { return HFFTWith(h, exN) })
		add("IHFFT", func() any { return IHFFT(r) }, func() any { return IHFFTWith(r, ex) })
		add("NewPlanWith", func() any { return NewPlan(n).FFT(make([]complex128, n), x) },
			func() any { return NewPlanWith(n, ex).FFT(make([]complex128, n), x) })
		add("NewPlanWith.IFFT", func() any { return NewPlan(n).IFFT(make([]complex128, n), x) },
			func() any { return NewPlanWith(n, ex).IFFT(make([]complex128, n), x) })
		add("NewRealPlanWith", func() any { return NewRealPlan(n).RFFT(make([]complex128, n/2+1), r) },
			func() any { return NewRealPlanWith(n, ex).RFFT(make([]complex128, n/2+1), r) })
		add("NewRealPlanWith.IRFFT", func() any { return NewRealPlan(n).IRFFT(make([]float64, n), h) },
			func() any { return NewRealPlanWith(n, ex).IRFFT(make([]float64, n), h) })
		// Options N, Norm: padded, truncated, every Norm.
		for _, norm := range []Norm{NormOrtho, NormForward} {
			pad := Options{N: n + 7, Norm: norm}
			cut := Options{N: n/2 + 1, Norm: norm}
			add("FFTWith/pad/"+norm.String(), func() any { return FFTWith(x, pad) }, func() any { return FFTWith(x, exact(pad)) })
			add("IFFTWith/cut/"+norm.String(), func() any { return IFFTWith(x, cut) }, func() any { return IFFTWith(x, exact(cut)) })
			add("RFFTWith/pad/"+norm.String(), func() any { return RFFTWith(r, pad) }, func() any { return RFFTWith(r, exact(pad)) })
		}
		add("FFT32With", func() any { return FFT32(x32) }, func() any { return FFT32With(x32, ex) })
		add("IFFT32With", func() any { return IFFT32(x32) }, func() any { return IFFT32With(x32, ex) })
		add("RFFT32With", func() any { return RFFT32(r32) }, func() any { return RFFT32With(r32, ex) })
		add("IRFFT32With", func() any { return IRFFT32(h32, n) }, func() any { return IRFFT32With(h32, exN) })
		add("HFFT32With", func() any { return HFFT32(h32, n) }, func() any { return HFFT32With(h32, exN) })
		add("IHFFT32With", func() any { return IHFFT32(r32) }, func() any { return IHFFT32With(r32, ex) })
		add("Plan32", func() any { return NewPlan32(n).FFT(make([]complex64, n), x32) }, func() any { return FFT32With(x32, ex) })
	}
	for _, sh := range shapes {
		key = fmt.Sprint("shape=", sh)
		size := shapeProduct(sh...)
		x := fmaSignal(size, uint64(size))
		r := fmaReal(size, uint64(size))
		sp := RFFTN(r, sh)
		x32 := make([]complex64, size)
		r32 := make([]float32, size)
		for i := range x {
			x32[i] = complex64(x[i])
			r32[i] = float32(r[i])
		}
		sp32 := RFFTN32(r32, sh)
		ex := exact(Options{})
		add("FFTN", func() any { return FFTN(x, sh) }, func() any { return FFTNWith(x, sh, ex) })
		add("IFFTN", func() any { return IFFTN(x, sh) }, func() any { return IFFTNWith(x, sh, ex) })
		add("RFFTN", func() any { return RFFTN(r, sh) }, func() any { return RFFTNWith(r, sh, ex) })
		add("IRFFTN", func() any { return IRFFTN(sp, sh) }, func() any { return IRFFTNWith(sp, sh, ex) })
		add("NewPlanNWith", func() any { return NewPlanN(sh...).FFT(make([]complex128, size), x) },
			func() any { return NewPlanNWith(sh, ex).FFT(make([]complex128, size), x) })
		add("NewPlanNWith.IFFT", func() any { return NewPlanN(sh...).IFFT(make([]complex128, size), x) },
			func() any { return NewPlanNWith(sh, ex).IFFT(make([]complex128, size), x) })
		for _, axes := range [][]int{{0}, {-1}} {
			o := Options{Axes: axes, Norm: NormOrtho}
			add(fmt.Sprint("FFTNWith/axes", axes), func() any { return FFTNWith(x, sh, o) }, func() any { return FFTNWith(x, sh, exact(o)) })
		}
		add("FFTN32With", func() any { return FFTN32(x32, sh) }, func() any { return FFTN32With(x32, sh, ex) })
		add("IFFTN32With", func() any { return IFFTN32(x32, sh) }, func() any { return IFFTN32With(x32, sh, ex) })
		add("RFFTN32With", func() any { return RFFTN32(r32, sh) }, func() any { return RFFTN32With(r32, sh, ex) })
		add("IRFFTN32With", func() any { return IRFFTN32(sp32, sh) }, func() any { return IRFFTN32With(sp32, sh, ex) })
		add("NewPlanN32", func() any { return NewPlanN32(sh...).FFT(make([]complex64, size), x32) }, func() any { return FFTN32With(x32, sh, ex) })
		if len(sh) == 2 {
			s2 := [2]int{sh[0], sh[1]}
			add("FFT2", func() any { return FFT2(x, s2) }, func() any { return FFT2With(x, s2, ex) })
			add("IFFT2", func() any { return IFFT2(x, s2) }, func() any { return IFFT2With(x, s2, ex) })
			add("RFFT2", func() any { return RFFT2(r, s2) }, func() any { return RFFT2With(r, s2, ex) })
			add("IRFFT2", func() any { return IRFFT2(sp, s2) }, func() any { return IRFFT2With(sp, s2, ex) })
			add("NewRealPlan2With", func() any { return NewRealPlan2(s2[0], s2[1]).RFFT(make([]complex128, s2[0]*(s2[1]/2+1)), r) },
				func() any { return NewRealPlan2With(s2[0], s2[1], ex).RFFT(make([]complex128, s2[0]*(s2[1]/2+1)), r) })
			add("NewRealPlan2With.IRFFT", func() any { return NewRealPlan2(s2[0], s2[1]).IRFFT(make([]float64, size), sp) },
				func() any { return NewRealPlan2With(s2[0], s2[1], ex).IRFFT(make([]float64, size), sp) })
			add("FFT2_32With", func() any { return FFT2_32(x32, s2) }, func() any { return FFT2_32With(x32, s2, ex) })
			add("IFFT2_32With", func() any { return IFFT2_32(x32, s2) }, func() any { return IFFT2_32With(x32, s2, ex) })
			add("RFFT2_32With", func() any { return RFFT2_32(r32, s2) }, func() any { return RFFT2_32With(r32, s2, ex) })
			add("IRFFT2_32With", func() any { return IRFFT2_32(sp32, s2) }, func() any { return IRFFT2_32With(sp32, s2, ex) })
		}
	}
	return cs
}

// digestAny is the SHA-256 of a result's bits (float bit patterns, NaNs
// canonicalized), as hex.
func digestAny(v any) string {
	h := sha256.New()
	var b [8]byte
	f64 := func(x float64) {
		if math.IsNaN(x) {
			x = math.NaN()
		}
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(x))
		h.Write(b[:])
	}
	f32 := func(x float32) {
		if x != x {
			x = float32(math.NaN())
		}
		binary.LittleEndian.PutUint32(b[:4], math.Float32bits(x))
		h.Write(b[:4])
	}
	switch x := v.(type) {
	case []complex128:
		for _, c := range x {
			f64(real(c))
			f64(imag(c))
		}
	case []float64:
		for _, c := range x {
			f64(c)
		}
	case []complex64:
		for _, c := range x {
			f32(real(c))
			f32(imag(c))
		}
	case []float32:
		for _, c := range x {
			f32(c)
		}
	default:
		panic(fmt.Sprintf("digestAny: %T", v))
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func sameAny(a, b any) bool { return digestAny(a) == digestAny(b) }
