package fft

// The measurements behind BENCHMARKS.md's Round 37 that run on every
// architecture: an RFFT split into its half transform and its untangle (and
// IRFFT the same way), rotated, and the export of inputs and outputs for the
// accuracy check against numpy.

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

var (
	hwParts     = flag.String("hw.parts", "", "rounds,ms: run TestHwParts")
	hwPartLens  = flag.String("hw.partlens", "256,1080,1024,1000,1920,4096", "TestHwParts' RFFT lengths")
	hwExport    = flag.String("hw.export", "", "TestHwExport: directory the inputs and outputs go to")
	hwExpReal   = flag.String("hw.expreal", "256,512,1000,1024,1080,1920,4096,65536", "TestHwExport: RFFT / IRFFT lengths")
	hwExpShapes = flag.String("hw.expshapes", "", "TestHwExport: N-D shapes (e.g. 1024x1024,4x256x1024)")
	hwExpComp   = flag.String("hw.expcomp", "", "TestHwExport: complex lengths")
)

// TestHwParts times, per RFFT length, RFFT, its half transform, the
// untangle, IRFFT and the retangle, each alone, in rotated rounds.
func TestHwParts(t *testing.T) {
	if *hwParts == "" {
		t.Skip("set -hw.parts=rounds,ms")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*hwParts, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	for _, s := range strings.Split(*hwPartLens, ",") {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		p := NewRealPlan(n)
		m := n / 2
		src := benchReal(n)
		zin := asComplex(src[:2*m])
		dst := make([]complex128, m+1)
		Z := make([]complex128, m)
		p.half.execute(Z, zin, false)
		p.RFFT(dst, src)
		back := make([]float64, n)
		zb := make([]complex128, m)
		h := 0.5 / float64(m)
		names := []string{"rfft", "half", "untangle", "irfft", "retangle"}
		fns := []func(){
			func() { p.RFFT(dst, src) },
			func() { p.half.execute(Z, zin, false) },
			func() { rfftUntangle(dst, Z, p.tw, p.ktw, m) },
			func() { p.IRFFT(back, dst) },
			func() { irfftRetangle(zb, dst, p.tw, p.ktw, m, h) },
		}
		times := smallRotate(fns, rounds, ms)
		var b strings.Builder
		for i, nm := range names {
			fmt.Fprintf(&b, " %s %.1f", nm, smallMedian(times[i]))
		}
		fmt.Printf("HWPARTS %d half(%s)%s\n", n, factorName(factorsOf(p.half)), b.String())
	}
}

// TestHwExport writes random inputs and go-fft's outputs as little-endian
// float64 into -hw.export, so numpy can judge them on the bytes this code
// received: RFFT and IRFFT at -hw.expreal's lengths, PlanN.FFT at
// -hw.expshapes' shapes and Plan.FFT at -hw.expcomp's lengths.
func TestHwExport(t *testing.T) {
	if *hwExport == "" {
		t.Skip("set -hw.export=dir")
	}
	r := rand.New(rand.NewPCG(37, 2026))
	write := func(name string, x []float64) {
		b := make([]byte, 0, 8*len(x))
		for _, v := range x {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
		}
		if err := os.WriteFile(filepath.Join(*hwExport, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cplx := func(x []complex128) []float64 { return unsafe.Slice((*float64)(unsafe.Pointer(&x[0])), 2*len(x)) }
	randC := func(n int) []complex128 {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		return x
	}
	list := func(s string) []string {
		if s == "" {
			return nil
		}
		return strings.Split(s, ",")
	}
	for _, s := range list(*hwExpReal) {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float64, n)
		for i := range x {
			x[i] = r.NormFloat64()
		}
		p := NewRealPlan(n)
		write(fmt.Sprintf("in_r_%d.bin", n), x)
		write(fmt.Sprintf("out_r_%d.bin", n), cplx(p.RFFT(make([]complex128, n/2+1), x)))
		spec := randC(n/2 + 1)
		spec[0], spec[n/2] = complex(real(spec[0]), 0), complex(real(spec[n/2]), 0)
		write(fmt.Sprintf("in_i_%d.bin", n), cplx(spec))
		write(fmt.Sprintf("out_i_%d.bin", n), p.IRFFT(make([]float64, n), spec))
		fmt.Printf("EXPORT r_%d half %s\n", n, factorName(factorsOf(p.half)))
	}
	for _, s := range list(*hwExpShapes) {
		var shape []int
		for _, x := range strings.Split(s, "x") {
			n, err := strconv.Atoi(x)
			if err != nil {
				t.Fatal(err)
			}
			shape = append(shape, n)
		}
		p := NewPlanN(shape...)
		x := randC(p.Len())
		write("in_nd_"+s+".bin", cplx(x))
		write("out_nd_"+s+".bin", cplx(p.FFT(make([]complex128, len(x)), x)))
		fmt.Printf("EXPORT nd_%s rows %s\n", s, factorName(factorsOf(p.axes[len(shape)-1])))
	}
	for _, s := range list(*hwExpComp) {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		p := NewPlan(n)
		x := randC(n)
		write(fmt.Sprintf("in_c_%d.bin", n), cplx(x))
		write(fmt.Sprintf("out_c_%d.bin", n), cplx(p.FFT(make([]complex128, n), x)))
		fmt.Printf("EXPORT c_%d factors %s\n", n, factorName(factorsOf(p)))
	}
}

// factorsOf lists a plan's Stockham radices (none for another engine).
func factorsOf(p *Plan) []int {
	var f []int
	if p.sk != nil {
		for _, st := range p.sk.stages {
			f = append(f, st.r)
		}
	}
	return f
}
