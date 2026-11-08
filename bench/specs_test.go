package bench

import (
	"fmt"

	fft "github.com/go-fft/fft"
)

const (
	tol64 = 1e-9 // relative to the largest reference value
	tol32 = 2e-4
)

var ortho = fft.Options{Norm: fft.NormOrtho}

// shapes: 1-D lengths (a power of two, a 5-smooth composite, a prime), a square
// matrix, a cube.
var (
	lens2d = [2]int{256, 256}
	cube   = []int{32, 32, 32}
	dctSh  = []int{128, 128}
)

const (
	nPow  = 1024
	nHalf = nPow/2 + 1
)

func name(f string, n int) string { return fmt.Sprintf("%s/%d", f, n) }

func shp(s []int) string { return fmt.Sprint(s) }

// specs builds every row. Plans are built here, once, outside the timed body:
// plan construction is excluded on both sides (scipy.fft's own plan cache is
// warm after the calibration call; see docs/perf.md).
func specs() []spec {
	var s []spec
	add := func(x ...spec) { s = append(s, x...) }

	// 1-D complex, float64: the one-shot functions and the plans, at three lengths.
	for _, n := range []int{nPow, 1000, 1009} {
		e, x := c128(n)
		in := []any{x}
		su := list(e)
		dst := make([]complex128, n)
		p := fft.NewPlan(n)
		add(
			mk(name("FFT", n), "FFT", su, py("lambda x: F.fft(x)"), tol64, in, func() []complex128 { return fft.FFT(x) }),
			mk(name("IFFT", n), "IFFT", su, py("lambda x: F.ifft(x)"), tol64, in, func() []complex128 { return fft.IFFT(x) }),
			mk(name("Plan.FFT", n), "Plan.FFT", su, py("lambda x: F.fft(x)"), tol64, in, func() []complex128 { return p.FFT(dst, x) }),
			mk(name("Plan.IFFT", n), "Plan.IFFT", su, py("lambda x: F.ifft(x)"), tol64, in, func() []complex128 { return p.IFFT(dst, x) }),
		)
		er, xr := r64(n)
		inr := []any{xr}
		rp := fft.NewRealPlan(n)
		rdst := make([]complex128, n/2+1)
		add(
			mk(name("RFFT", n), "RFFT", list(er), py("lambda x: F.rfft(x)"), tol64, inr, func() []complex128 { return fft.RFFT(xr) }),
			mk(name("RealPlan.RFFT", n), "RealPlan.RFFT", list(er), py("lambda x: F.rfft(x)"), tol64, inr, func() []complex128 { return rp.RFFT(rdst, xr) }),
		)
		if n%2 == 0 {
			eh, h := h128(n/2 + 1)
			inh := []any{h}
			odst := make([]float64, n)
			add(
				mk(name("IRFFT", n), "IRFFT", list(eh), py(fmt.Sprintf("lambda x: F.irfft(x, %d)", n)), tol64, inh, func() []float64 { return fft.IRFFT(h, n) }),
				mk(name("RealPlan.IRFFT", n), "RealPlan.IRFFT", list(eh), py(fmt.Sprintf("lambda x: F.irfft(x, %d)", n)), tol64, inh, func() []float64 { return rp.IRFFT(odst, h) }),
			)
		}
	}

	// The remaining 1-D float64 functions at n = 1024.
	{
		e, x := c128(nPow)
		in := []any{x}
		su := list(e)
		dst := make([]complex128, nPow)
		p := fft.NewPlan(nPow)
		er, xr := r64(nPow)
		inr := []any{xr}
		eh, h := h128(nHalf)
		inh := []any{h}
		rp := fft.NewRealPlan(nPow)
		rdst := make([]complex128, nHalf)
		odst := make([]float64, nPow)
		add(
			mk("FFTWith", "FFTWith", su, py("lambda x: F.fft(x, norm='ortho')"), tol64, in, func() []complex128 { return fft.FFTWith(x, ortho) }),
			mk("IFFTWith", "IFFTWith", su, py("lambda x: F.ifft(x, norm='ortho')"), tol64, in, func() []complex128 { return fft.IFFTWith(x, ortho) }),
			mk("Plan.FFTNorm", "Plan.FFTNorm", su, py("lambda x: F.fft(x, norm='ortho')"), tol64, in, func() []complex128 { return p.FFTNorm(dst, x, fft.NormOrtho) }),
			mk("Plan.IFFTNorm", "Plan.IFFTNorm", su, py("lambda x: F.ifft(x, norm='ortho')"), tol64, in, func() []complex128 { return p.IFFTNorm(dst, x, fft.NormOrtho) }),
			mk("FFTReal", "FFTReal", list(er), py("lambda x: F.fft(x)"), tol64, inr, func() []complex128 { return fft.FFTReal(xr) }),
			mk("RFFTWith", "RFFTWith", list(er), py("lambda x: F.rfft(x, norm='ortho')"), tol64, inr, func() []complex128 { return fft.RFFTWith(xr, ortho) }),
			mk("RealPlan.RFFTNorm", "RealPlan.RFFTNorm", list(er), py("lambda x: F.rfft(x, norm='ortho')"), tol64, inr, func() []complex128 { return rp.RFFTNorm(rdst, xr, fft.NormOrtho) }),
			mk("IRFFTWith", "IRFFTWith", list(eh), py("lambda x: F.irfft(x, norm='ortho')"), tol64, inh, func() []float64 { return fft.IRFFTWith(h, ortho) }),
			mk("RealPlan.IRFFTNorm", "RealPlan.IRFFTNorm", list(eh), py("lambda x: F.irfft(x, 1024, norm='ortho')"), tol64, inh, func() []float64 { return rp.IRFFTNorm(odst, h, fft.NormOrtho) }),
			mk("HFFT", "HFFT", list(eh), py("lambda x: F.hfft(x, 1024)"), tol64, inh, func() []float64 { return fft.HFFT(h, nPow) }),
			mk("HFFTWith", "HFFTWith", list(eh), py("lambda x: F.hfft(x, norm='ortho')"), tol64, inh, func() []float64 { return fft.HFFTWith(h, ortho) }),
			mk("IHFFT", "IHFFT", list(er), py("lambda x: F.ihfft(x)"), tol64, inr, func() []complex128 { return fft.IHFFT(xr) }),
			mk("IHFFTWith", "IHFFTWith", list(er), py("lambda x: F.ihfft(x, norm='ortho')"), tol64, inr, func() []complex128 { return fft.IHFFTWith(xr, ortho) }),
		)
	}

	// 1-D float32 (scipy.fft computes in single precision on float32/complex64).
	{
		e, x := c64(nPow)
		in := []any{x}
		su := list(e)
		dst := make([]complex64, nPow)
		p := fft.NewPlan32(nPow)
		er, xr := r32(nPow)
		inr := []any{xr}
		eh, h := h64(nHalf)
		inh := []any{h}
		rp := fft.NewRealPlan32(nPow)
		rdst := make([]complex64, nHalf)
		odst := make([]float32, nPow)
		add(
			mk("FFT32", "FFT32", su, py("lambda x: F.fft(x)"), tol32, in, func() []complex64 { return fft.FFT32(x) }),
			mk("IFFT32", "IFFT32", su, py("lambda x: F.ifft(x)"), tol32, in, func() []complex64 { return fft.IFFT32(x) }),
			mk("FFT32With", "FFT32With", su, py("lambda x: F.fft(x, norm='ortho')"), tol32, in, func() []complex64 { return fft.FFT32With(x, ortho) }),
			mk("IFFT32With", "IFFT32With", su, py("lambda x: F.ifft(x, norm='ortho')"), tol32, in, func() []complex64 { return fft.IFFT32With(x, ortho) }),
			mk("Plan32.FFT", "Plan32.FFT", su, py("lambda x: F.fft(x)"), tol32, in, func() []complex64 { return p.FFT(dst, x) }),
			mk("Plan32.IFFT", "Plan32.IFFT", su, py("lambda x: F.ifft(x)"), tol32, in, func() []complex64 { return p.IFFT(dst, x) }),
			mk("Plan32.FFTNorm", "Plan32.FFTNorm", su, py("lambda x: F.fft(x, norm='ortho')"), tol32, in, func() []complex64 { return p.FFTNorm(dst, x, fft.NormOrtho) }),
			mk("Plan32.IFFTNorm", "Plan32.IFFTNorm", su, py("lambda x: F.ifft(x, norm='ortho')"), tol32, in, func() []complex64 { return p.IFFTNorm(dst, x, fft.NormOrtho) }),
			mk("RFFT32", "RFFT32", list(er), py("lambda x: F.rfft(x)"), tol32, inr, func() []complex64 { return fft.RFFT32(xr) }),
			mk("RFFT32With", "RFFT32With", list(er), py("lambda x: F.rfft(x, norm='ortho')"), tol32, inr, func() []complex64 { return fft.RFFT32With(xr, ortho) }),
			mk("RealPlan32.RFFT", "RealPlan32.RFFT", list(er), py("lambda x: F.rfft(x)"), tol32, inr, func() []complex64 { return rp.RFFT(rdst, xr) }),
			mk("RealPlan32.RFFTNorm", "RealPlan32.RFFTNorm", list(er), py("lambda x: F.rfft(x, norm='ortho')"), tol32, inr, func() []complex64 { return rp.RFFTNorm(rdst, xr, fft.NormOrtho) }),
			mk("IRFFT32", "IRFFT32", list(eh), py("lambda x: F.irfft(x, 1024)"), tol32, inh, func() []float32 { return fft.IRFFT32(h, nPow) }),
			mk("IRFFT32With", "IRFFT32With", list(eh), py("lambda x: F.irfft(x, norm='ortho')"), tol32, inh, func() []float32 { return fft.IRFFT32With(h, ortho) }),
			mk("RealPlan32.IRFFT", "RealPlan32.IRFFT", list(eh), py("lambda x: F.irfft(x, 1024)"), tol32, inh, func() []float32 { return rp.IRFFT(odst, h) }),
			mk("RealPlan32.IRFFTNorm", "RealPlan32.IRFFTNorm", list(eh), py("lambda x: F.irfft(x, 1024, norm='ortho')"), tol32, inh, func() []float32 { return rp.IRFFTNorm(odst, h, fft.NormOrtho) }),
			mk("HFFT32", "HFFT32", list(eh), py("lambda x: F.hfft(x, 1024)"), tol32, inh, func() []float32 { return fft.HFFT32(h, nPow) }),
			mk("HFFT32With", "HFFT32With", list(eh), py("lambda x: F.hfft(x, norm='ortho')"), tol32, inh, func() []float32 { return fft.HFFT32With(h, ortho) }),
			mk("IHFFT32", "IHFFT32", list(er), py("lambda x: F.ihfft(x)"), tol32, inr, func() []complex64 { return fft.IHFFT32(xr) }),
			mk("IHFFT32With", "IHFFT32With", list(er), py("lambda x: F.ihfft(x, norm='ortho')"), tol32, inr, func() []complex64 { return fft.IHFFT32With(xr, ortho) }),
		)
	}

	// 2-D, 256x256.
	{
		r, c := lens2d[0], lens2d[1]
		N := r * c
		sh := []int{r, c}
		e, x := c128(N, r, c)
		in := []any{x}
		su := list(e)
		dst := make([]complex128, N)
		pn := fft.NewPlanN(r, c)
		er, xr := r64(N, r, c)
		inr := []any{xr}
		eh, h := c128(r*(c/2+1), r, c/2+1)
		inh := []any{h}
		rp := fft.NewRealPlan2(r, c)
		rdst := make([]complex128, rp.SpectrumLen())
		odst := make([]float64, N)
		irf := fmt.Sprintf("s=(%d, %d)", r, c)
		add(
			mk("FFT2", "FFT2", su, py("lambda x: F.fft2(x)"), tol64, in, func() []complex128 { return fft.FFT2(x, lens2d) }),
			mk("FFT2With", "FFT2With", su, py("lambda x: F.fft2(x, norm='ortho')"), tol64, in, func() []complex128 { return fft.FFT2With(x, lens2d, ortho) }),
			mk("IFFT2", "IFFT2", su, py("lambda x: F.ifft2(x)"), tol64, in, func() []complex128 { return fft.IFFT2(x, lens2d) }),
			mk("IFFT2With", "IFFT2With", su, py("lambda x: F.ifft2(x, norm='ortho')"), tol64, in, func() []complex128 { return fft.IFFT2With(x, lens2d, ortho) }),
			mk("PlanN.FFT", "PlanN.FFT", su, py("lambda x: F.fft2(x)"), tol64, in, func() []complex128 { return pn.FFT(dst, x) }),
			mk("PlanN.IFFT", "PlanN.IFFT", su, py("lambda x: F.ifft2(x)"), tol64, in, func() []complex128 { return pn.IFFT(dst, x) }),
			mk("PlanN.FFTNorm", "PlanN.FFTNorm", su, py("lambda x: F.fft2(x, norm='ortho')"), tol64, in, func() []complex128 { return pn.FFTNorm(dst, x, fft.NormOrtho) }),
			mk("PlanN.IFFTNorm", "PlanN.IFFTNorm", su, py("lambda x: F.ifft2(x, norm='ortho')"), tol64, in, func() []complex128 { return pn.IFFTNorm(dst, x, fft.NormOrtho) }),
			mk("RFFT2", "RFFT2", list(er), py("lambda x: F.rfft2(x)"), tol64, inr, func() []complex128 { return fft.RFFT2(xr, lens2d) }),
			mk("RFFT2With", "RFFT2With", list(er), py("lambda x: F.rfft2(x, norm='ortho')"), tol64, inr, func() []complex128 { return fft.RFFT2With(xr, lens2d, ortho) }),
			mk("RealPlan2.RFFT", "RealPlan2.RFFT", list(er), py("lambda x: F.rfft2(x)"), tol64, inr, func() []complex128 { return rp.RFFT(rdst, xr) }),
			mk("RealPlan2.RFFTNorm", "RealPlan2.RFFTNorm", list(er), py("lambda x: F.rfft2(x, norm='ortho')"), tol64, inr, func() []complex128 { return rp.RFFTNorm(rdst, xr, fft.NormOrtho) }),
			mk("IRFFT2", "IRFFT2", list(eh), py("lambda x: F.irfft2(x, "+irf+")"), tol64, inh, func() []float64 { return fft.IRFFT2(h, lens2d) }),
			mk("IRFFT2With", "IRFFT2With", list(eh), py("lambda x: F.irfft2(x, "+irf+", norm='ortho')"), tol64, inh, func() []float64 { return fft.IRFFT2With(h, lens2d, ortho) }),
			mk("RealPlan2.IRFFT", "RealPlan2.IRFFT", list(eh), py("lambda x: F.irfft2(x, "+irf+")"), tol64, inh, func() []float64 { return rp.IRFFT(odst, h) }),
			mk("RealPlan2.IRFFTNorm", "RealPlan2.IRFFTNorm", list(eh), py("lambda x: F.irfft2(x, "+irf+", norm='ortho')"), tol64, inh, func() []float64 { return rp.IRFFTNorm(odst, h, fft.NormOrtho) }),
		)
		_ = sh
	}

	// 2-D, float32.
	{
		r, c := lens2d[0], lens2d[1]
		N := r * c
		e, x := c64(N, r, c)
		in := []any{x}
		su := list(e)
		dst := make([]complex64, N)
		pn := fft.NewPlanN32(r, c)
		er, xr := r32(N, r, c)
		inr := []any{xr}
		eh, h := c64(r*(c/2+1), r, c/2+1)
		inh := []any{h}
		rp := fft.NewRealPlan2_32(r, c)
		rdst := make([]complex64, rp.SpectrumLen())
		odst := make([]float32, N)
		irf := fmt.Sprintf("s=(%d, %d)", r, c)
		add(
			mk("FFT2_32", "FFT2_32", su, py("lambda x: F.fft2(x)"), tol32, in, func() []complex64 { return fft.FFT2_32(x, lens2d) }),
			mk("FFT2_32With", "FFT2_32With", su, py("lambda x: F.fft2(x, norm='ortho')"), tol32, in, func() []complex64 { return fft.FFT2_32With(x, lens2d, ortho) }),
			mk("IFFT2_32", "IFFT2_32", su, py("lambda x: F.ifft2(x)"), tol32, in, func() []complex64 { return fft.IFFT2_32(x, lens2d) }),
			mk("IFFT2_32With", "IFFT2_32With", su, py("lambda x: F.ifft2(x, norm='ortho')"), tol32, in, func() []complex64 { return fft.IFFT2_32With(x, lens2d, ortho) }),
			mk("PlanN32.FFT", "PlanN32.FFT", su, py("lambda x: F.fft2(x)"), tol32, in, func() []complex64 { return pn.FFT(dst, x) }),
			mk("PlanN32.IFFT", "PlanN32.IFFT", su, py("lambda x: F.ifft2(x)"), tol32, in, func() []complex64 { return pn.IFFT(dst, x) }),
			mk("PlanN32.FFTNorm", "PlanN32.FFTNorm", su, py("lambda x: F.fft2(x, norm='ortho')"), tol32, in, func() []complex64 { return pn.FFTNorm(dst, x, fft.NormOrtho) }),
			mk("PlanN32.IFFTNorm", "PlanN32.IFFTNorm", su, py("lambda x: F.ifft2(x, norm='ortho')"), tol32, in, func() []complex64 { return pn.IFFTNorm(dst, x, fft.NormOrtho) }),
			mk("RFFT2_32", "RFFT2_32", list(er), py("lambda x: F.rfft2(x)"), tol32, inr, func() []complex64 { return fft.RFFT2_32(xr, lens2d) }),
			mk("RFFT2_32With", "RFFT2_32With", list(er), py("lambda x: F.rfft2(x, norm='ortho')"), tol32, inr, func() []complex64 { return fft.RFFT2_32With(xr, lens2d, ortho) }),
			mk("RealPlan2_32.RFFT", "RealPlan2_32.RFFT", list(er), py("lambda x: F.rfft2(x)"), tol32, inr, func() []complex64 { return rp.RFFT(rdst, xr) }),
			mk("RealPlan2_32.RFFTNorm", "RealPlan2_32.RFFTNorm", list(er), py("lambda x: F.rfft2(x, norm='ortho')"), tol32, inr, func() []complex64 { return rp.RFFTNorm(rdst, xr, fft.NormOrtho) }),
			mk("IRFFT2_32", "IRFFT2_32", list(eh), py("lambda x: F.irfft2(x, "+irf+")"), tol32, inh, func() []float32 { return fft.IRFFT2_32(h, lens2d) }),
			mk("IRFFT2_32With", "IRFFT2_32With", list(eh), py("lambda x: F.irfft2(x, "+irf+", norm='ortho')"), tol32, inh, func() []float32 { return fft.IRFFT2_32With(h, lens2d, ortho) }),
			mk("RealPlan2_32.IRFFT", "RealPlan2_32.IRFFT", list(eh), py("lambda x: F.irfft2(x, "+irf+")"), tol32, inh, func() []float32 { return rp.IRFFT(odst, h) }),
			mk("RealPlan2_32.IRFFTNorm", "RealPlan2_32.IRFFTNorm", list(eh), py("lambda x: F.irfft2(x, "+irf+", norm='ortho')"), tol32, inh, func() []float32 { return rp.IRFFTNorm(odst, h, fft.NormOrtho) }),
		)
	}

	// N-D, 32x32x32, float64 and float32.
	{
		N := prod(cube)
		half := []int{32, 32, 17}
		irf := "s=(32, 32, 32)"
		e, x := c128(N, cube...)
		in := []any{x}
		su := list(e)
		er, xr := r64(N, cube...)
		inr := []any{xr}
		eh, h := c128(prod(half), half...)
		inh := []any{h}
		e3, x3 := c64(N, cube...)
		in3 := []any{x3}
		er3, xr3 := r32(N, cube...)
		inr3 := []any{xr3}
		eh3, h3 := c64(prod(half), half...)
		inh3 := []any{h3}
		add(
			mk("FFTN", "FFTN", su, py("lambda x: F.fftn(x)"), tol64, in, func() []complex128 { return fft.FFTN(x, cube) }),
			mk("FFTNWith", "FFTNWith", su, py("lambda x: F.fftn(x, norm='ortho')"), tol64, in, func() []complex128 { return fft.FFTNWith(x, cube, ortho) }),
			mk("IFFTN", "IFFTN", su, py("lambda x: F.ifftn(x)"), tol64, in, func() []complex128 { return fft.IFFTN(x, cube) }),
			mk("IFFTNWith", "IFFTNWith", su, py("lambda x: F.ifftn(x, norm='ortho')"), tol64, in, func() []complex128 { return fft.IFFTNWith(x, cube, ortho) }),
			mk("RFFTN", "RFFTN", list(er), py("lambda x: F.rfftn(x)"), tol64, inr, func() []complex128 { return fft.RFFTN(xr, cube) }),
			mk("RFFTNWith", "RFFTNWith", list(er), py("lambda x: F.rfftn(x, norm='ortho')"), tol64, inr, func() []complex128 { return fft.RFFTNWith(xr, cube, ortho) }),
			mk("IRFFTN", "IRFFTN", list(eh), py("lambda x: F.irfftn(x, "+irf+")"), tol64, inh, func() []float64 { return fft.IRFFTN(h, cube) }),
			mk("IRFFTNWith", "IRFFTNWith", list(eh), py("lambda x: F.irfftn(x, "+irf+", norm='ortho')"), tol64, inh, func() []float64 { return fft.IRFFTNWith(h, cube, ortho) }),
			mk("FFTN32", "FFTN32", list(e3), py("lambda x: F.fftn(x)"), tol32, in3, func() []complex64 { return fft.FFTN32(x3, cube) }),
			mk("FFTN32With", "FFTN32With", list(e3), py("lambda x: F.fftn(x, norm='ortho')"), tol32, in3, func() []complex64 { return fft.FFTN32With(x3, cube, ortho) }),
			mk("IFFTN32", "IFFTN32", list(e3), py("lambda x: F.ifftn(x)"), tol32, in3, func() []complex64 { return fft.IFFTN32(x3, cube) }),
			mk("IFFTN32With", "IFFTN32With", list(e3), py("lambda x: F.ifftn(x, norm='ortho')"), tol32, in3, func() []complex64 { return fft.IFFTN32With(x3, cube, ortho) }),
			mk("RFFTN32", "RFFTN32", list(er3), py("lambda x: F.rfftn(x)"), tol32, inr3, func() []complex64 { return fft.RFFTN32(xr3, cube) }),
			mk("RFFTN32With", "RFFTN32With", list(er3), py("lambda x: F.rfftn(x, norm='ortho')"), tol32, inr3, func() []complex64 { return fft.RFFTN32With(xr3, cube, ortho) }),
			mk("IRFFTN32", "IRFFTN32", list(eh3), py("lambda x: F.irfftn(x, "+irf+")"), tol32, inh3, func() []float32 { return fft.IRFFTN32(h3, cube) }),
			mk("IRFFTN32With", "IRFFTN32With", list(eh3), py("lambda x: F.irfftn(x, "+irf+", norm='ortho')"), tol32, inh3, func() []float32 { return fft.IRFFTN32With(h3, cube, ortho) }),
		)
	}

	// Real-to-real: type 2, orthonormal, at 1024 points and 128x128.
	{
		er, x := r64(nPow)
		er3, x3 := r32(nPow)
		en, xn := r64(prod(dctSh), dctSh...)
		in, in3, inn := []any{x}, []any{x3}, []any{xn}
		dp, ip := fft.NewDCTPlan(nPow, 2), fft.NewDSTPlan(nPow, 2)
		dp3, ip3 := fft.NewDCTPlan32(nPow, 2), fft.NewDSTPlan32(nPow, 2)
		dst, dst3 := make([]float64, nPow), make([]float32, nPow)
		no := fft.NormOrtho
		sh := "(128, 128)"
		add(
			mk("DCT", "DCT", list(er), py("lambda x: F.dct(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return fft.DCT(x, 2, no) }),
			mk("IDCT", "IDCT", list(er), py("lambda x: F.idct(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return fft.IDCT(x, 2, no) }),
			mk("DST", "DST", list(er), py("lambda x: F.dst(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return fft.DST(x, 2, no) }),
			mk("IDST", "IDST", list(er), py("lambda x: F.idst(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return fft.IDST(x, 2, no) }),
			mk("DCTPlan.DCT", "DCTPlan.DCT", list(er), py("lambda x: F.dct(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return dp.DCT(dst, x, no) }),
			mk("DCTPlan.IDCT", "DCTPlan.IDCT", list(er), py("lambda x: F.idct(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return dp.IDCT(dst, x, no) }),
			mk("DSTPlan.DST", "DSTPlan.DST", list(er), py("lambda x: F.dst(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return ip.DST(dst, x, no) }),
			mk("DSTPlan.IDST", "DSTPlan.IDST", list(er), py("lambda x: F.idst(x, type=2, norm='ortho')"), tol64, in, func() []float64 { return ip.IDST(dst, x, no) }),
			mk("DCT32", "DCT32", list(er3), py("lambda x: F.dct(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return fft.DCT32(x3, 2, no) }),
			mk("IDCT32", "IDCT32", list(er3), py("lambda x: F.idct(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return fft.IDCT32(x3, 2, no) }),
			mk("DST32", "DST32", list(er3), py("lambda x: F.dst(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return fft.DST32(x3, 2, no) }),
			mk("IDST32", "IDST32", list(er3), py("lambda x: F.idst(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return fft.IDST32(x3, 2, no) }),
			mk("DCTPlan32.DCT", "DCTPlan32.DCT", list(er3), py("lambda x: F.dct(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return dp3.DCT(dst3, x3, no) }),
			mk("DCTPlan32.IDCT", "DCTPlan32.IDCT", list(er3), py("lambda x: F.idct(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return dp3.IDCT(dst3, x3, no) }),
			mk("DSTPlan32.DST", "DSTPlan32.DST", list(er3), py("lambda x: F.dst(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return ip3.DST(dst3, x3, no) }),
			mk("DSTPlan32.IDST", "DSTPlan32.IDST", list(er3), py("lambda x: F.idst(x, type=2, norm='ortho')"), tol32, in3, func() []float32 { return ip3.IDST(dst3, x3, no) }),
			mk("DCTN", "DCTN", list(en), py("lambda x: F.dctn(x, type=2, norm='ortho')"), tol64, inn, func() []float64 { return fft.DCTN(xn, dctSh, 2, no) }),
			mk("IDCTN", "IDCTN", list(en), py("lambda x: F.idctn(x, type=2, norm='ortho')"), tol64, inn, func() []float64 { return fft.IDCTN(xn, dctSh, 2, no) }),
			mk("DSTN", "DSTN", list(en), py("lambda x: F.dstn(x, type=2, norm='ortho')"), tol64, inn, func() []float64 { return fft.DSTN(xn, dctSh, 2, no) }),
			mk("IDSTN", "IDSTN", list(en), py("lambda x: F.idstn(x, type=2, norm='ortho')"), tol64, inn, func() []float64 { return fft.IDSTN(xn, dctSh, 2, no) }),
		)
		_ = sh
		e32, xn32 := r32(prod(dctSh), dctSh...)
		in32 := []any{xn32}
		add(
			mk("DCTN32", "DCTN32", list(e32), py("lambda x: F.dctn(x, type=2, norm='ortho')"), tol32, in32, func() []float32 { return fft.DCTN32(xn32, dctSh, 2, no) }),
			mk("IDCTN32", "IDCTN32", list(e32), py("lambda x: F.idctn(x, type=2, norm='ortho')"), tol32, in32, func() []float32 { return fft.IDCTN32(xn32, dctSh, 2, no) }),
			mk("DSTN32", "DSTN32", list(e32), py("lambda x: F.dstn(x, type=2, norm='ortho')"), tol32, in32, func() []float32 { return fft.DSTN32(xn32, dctSh, 2, no) }),
			mk("IDSTN32", "IDSTN32", list(e32), py("lambda x: F.idstn(x, type=2, norm='ortho')"), tol32, in32, func() []float32 { return fft.IDSTN32(xn32, dctSh, 2, no) }),
		)
	}

	// Frequencies, shifts, windows, spectra.
	{
		er, x := r64(nPow)
		in := []any{x}
		e2, x2 := r64(prod(lens2d[:]), lens2d[0], lens2d[1])
		in2 := []any{x2}
		sh2 := lens2d[:]
		const nx, seg, ov = 16384, 256, 128
		ex, sig := r64(nx)
		ew := "(" + pyRe(seg) + " + 0.5)"
		win := make([]float64, seg)
		for i, v := range reVec(seg) {
			win[i] = v + 0.5
		}
		spec := "lambda x, w: (lambda W: (np.abs(F.rfft(W * w, axis=1)) ** 2) * (np.r_[1.0, np.full(127, 2.0), 1.0] / (1000.0 * 256)))(np.lib.stride_tricks.sliding_window_view(x, 256)[::128])"
		add(
			mk("FFTFreq", "FFTFreq", "[]", py("lambda: F.fftfreq(1024, 0.001)"), tol64, nil, func() []float64 { return fft.FFTFreq(nPow, 0.001) }),
			mk("RFFTFreq", "RFFTFreq", "[]", py("lambda: F.rfftfreq(1024, 0.001)"), tol64, nil, func() []float64 { return fft.RFFTFreq(nPow, 0.001) }),
			mk("FFTShift", "FFTShift", list(er), py("lambda x: F.fftshift(x)"), tol64, in, func() []float64 { return fft.FFTShift(x) }),
			mk("IFFTShift", "IFFTShift", list(er), py("lambda x: F.ifftshift(x)"), tol64, in, func() []float64 { return fft.IFFTShift(x) }),
			mk("FFTShiftN", "FFTShiftN", list(e2), py("lambda x: F.fftshift(x)"), tol64, in2, func() []float64 { return fft.FFTShiftN(x2, sh2, nil) }),
			mk("IFFTShiftN", "IFFTShiftN", list(e2), py("lambda x: F.ifftshift(x)"), tol64, in2, func() []float64 { return fft.IFFTShiftN(x2, sh2, nil) }),
			// NextFastLen's rule differs from scipy's on purpose (its doc): timing only.
			mk("NextFastLen", "NextFastLen", "[]", py("lambda: F.next_fast_len(1031, False)"), -1, nil, func() int { return fft.NextFastLen(1031, false) }),
			mk("Hann", "Hann", "[]", "lambda: np.hanning(1024)", tol64, nil, func() []float64 { return fft.Hann(nPow) }),
			mk("Hamming", "Hamming", "[]", "lambda: np.hamming(1024)", tol64, nil, func() []float64 { return fft.Hamming(nPow) }),
			mk("Blackman", "Blackman", "[]", "lambda: np.blackman(1024)", tol64, nil, func() []float64 { return fft.Blackman(nPow) }),
			mk("Bartlett", "Bartlett", "[]", "lambda: np.bartlett(1024)", tol64, nil, func() []float64 { return fft.Bartlett(nPow) }),
			mk("BlackmanHarris", "BlackmanHarris", "[]", pysig("lambda: S.windows.blackmanharris(1024, sym=True)"), tol64, nil, func() []float64 { return fft.BlackmanHarris(nPow) }),
			mk("PSD", "PSD", list(er), pysig("lambda x: S.periodogram(x, 1 / 0.001, window='boxcar', detrend=False, scaling='density')[1]"), tol64, in, func() []float64 { return fft.PSD(x, 0.001) }),
			mk("Spectrogram", "Spectrogram", list(ex, ew), py(spec), tol64, []any{sig, win}, func() [][]float64 { return fft.Spectrogram(sig, seg, ov, win, 0.001) }),
		)
	}
	return s
}
