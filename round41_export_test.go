package fft

import (
	"encoding/binary"
	"flag"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// Round 41's accuracy export: random inputs and the default and FMA plans'
// outputs as raw little-endian float64, for scripts/r41_accuracy.py to hold
// against numpy on the same bytes.

var r41Export = flag.String("r41.export", "", "directory: run TestR41Export (writes there, outside any repository)")

func r41Write(t *testing.T, dir, name string, v []float64) {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(x))
	}
	if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func r41Floats(c []complex128) []float64 {
	f := make([]float64, 0, 2*len(c))
	for _, z := range c {
		f = append(f, real(z), imag(z))
	}
	return f
}

// TestR41Export writes, for every length below, in_<kind>_<n>.bin and
// def_ and fma_ outputs: c (FFT), ic (IFFT), r (RFFT), ir (IRFFT of the
// input's RFFT, numpy's irfft of the same spectrum bytes), n (2-D FFT).
func TestR41Export(t *testing.T) {
	dir := *r41Export
	if dir == "" {
		t.Skip("set -r41.export=DIR")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Logf("fmaAvailable=%v", fmaAvailable)
	rng := rand.New(rand.NewPCG(41, 41))
	cin := func(n int) []complex128 {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(2*rng.Float64()-1, 2*rng.Float64()-1)
		}
		return x
	}
	var complexLens []int
	for e := 4; e <= 20; e++ {
		complexLens = append(complexLens, 1<<e)
	}
	complexLens = append(complexLens, 100, 1000, 1080, 1296, 1920, 6000, 20160, 45000, 100000, 720720, // composites
		1009, 10007, 65537, 999983, 1000003) // primes: Rader and Bluestein
	for _, n := range complexLens {
		x := cin(n)
		r41Write(t, dir, "in_c_"+strconv.Itoa(n)+".bin", r41Floats(x))
		for v, fma := range []bool{false, true} {
			p := newPlanF(n, fma)
			tag := [2]string{"def_", "fma_"}[v]
			r41Write(t, dir, tag+"c_"+strconv.Itoa(n)+".bin", r41Floats(p.FFT(make([]complex128, n), x)))
			r41Write(t, dir, tag+"ic_"+strconv.Itoa(n)+".bin", r41Floats(p.IFFT(make([]complex128, n), x)))
		}
	}
	realLens := []int{256, 1000, 1024, 1080, 1920, 4096, 65536, 1 << 20, 1001, 20000}
	for _, n := range realLens {
		x := make([]float64, n)
		for i := range x {
			x[i] = 2*rng.Float64() - 1
		}
		r41Write(t, dir, "in_r_"+strconv.Itoa(n)+".bin", x)
		spec := NewRealPlan(n).RFFT(make([]complex128, n/2+1), x)
		r41Write(t, dir, "in_ir_"+strconv.Itoa(n)+".bin", r41Floats(spec))
		for v, fma := range []bool{false, true} {
			p := newRealPlanF(n, fma)
			tag := [2]string{"def_", "fma_"}[v]
			r41Write(t, dir, tag+"r_"+strconv.Itoa(n)+".bin", r41Floats(p.RFFT(make([]complex128, n/2+1), x)))
			r41Write(t, dir, tag+"ir_"+strconv.Itoa(n)+".bin", p.IRFFT(make([]float64, n), spec))
		}
	}
	for _, sh := range [][2]int{{64, 64}, {256, 256}, {512, 512}, {1024, 1024}, {171, 64}, {100, 360}} {
		n := sh[0] * sh[1]
		x := cin(n)
		name := strconv.Itoa(sh[0]) + "x" + strconv.Itoa(sh[1])
		r41Write(t, dir, "in_n_"+name+".bin", r41Floats(x))
		for v, fma := range []bool{false, true} {
			p := newPlanNF([]int{sh[0], sh[1]}, fma)
			tag := [2]string{"def_", "fma_"}[v]
			r41Write(t, dir, tag+"n_"+name+".bin", r41Floats(p.FFT(make([]complex128, n), x)))
		}
	}
}
