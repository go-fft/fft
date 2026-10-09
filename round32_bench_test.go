package fft

// The in-process measurements behind BENCHMARKS.md's Round 32 (Zen 3, the
// smallest sizes): each variant timed in rotated rounds in one process.

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

	"github.com/go-fft/fft/internal/kernels"
)

var (
	znAB   = flag.String("zn.ab", "", "rounds,ms: run Round 32's in-process tests")
	znRows = flag.String("zn.rows", "C16,C32,C64,C128,C256,R64,R128,R256,R512,I256", "TestZnStack's rows: C complex, R RFFT, I IRFFT")
)

func znRounds(t *testing.T) (rounds, ms int) {
	if *znAB == "" {
		t.Skip("set -zn.ab=rounds,ms")
	}
	if _, err := fmt.Sscanf(*znAB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	return rounds, ms
}

// znRow returns the call a row of -zn.rows names.
func znRow(t *testing.T, row string) func() {
	n, err := strconv.Atoi(row[1:])
	if err != nil {
		t.Fatal(err)
	}
	switch row[0] {
	case 'C':
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		return func() { p.FFT(dst, src) }
	case 'R':
		znForget(n / 2)
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		return func() { p.RFFT(dst, src) }
	case 'I':
		znForget(n / 2)
		p := NewRealPlan(n)
		spec := p.RFFT(make([]complex128, n/2+1), benchReal(n))
		back := make([]float64, n)
		return func() { p.IRFFT(back, spec) }
	}
	t.Fatalf("row %q", row)
	return nil
}

// TestZnFrame times every row with the buffers between the passes from the
// plan's pool (plans built with the frame kernels off) and in the frame
// kernels' stack frame (built with them at the kernels package's limits),
// rotated, in one process.
func TestZnFrame(t *testing.T) {
	rounds, ms := znRounds(t)
	defer znSetMax(0, 0)()
	for _, row := range strings.Split(*znRows, ",") {
		znSetMax(0, 0)
		pool := znRow(t, row)
		znSetMax(kernels.ZnTwoPassMax, kernels.ZnThreePassMax)
		frame := znRow(t, row)
		times := smallRotate([]func(){pool, frame}, rounds, ms)
		fmt.Printf("ZNTWO %-5s pool %7.1f frame %7.1f pool/frame %s\n", row,
			smallMedian(times[0]), smallMedian(times[1]), smallRatio(times[0], times[1]))
	}
}

// TestZnFrameLengths lists the lengths up to 512 whose plans have two or
// three Stockham passes on this machine, their radices and layouts, and
// whether they take a frame kernel.
func TestZnFrameLengths(t *testing.T) {
	znRounds(t)
	for n := 2; n <= 512; n++ {
		if p := NewPlan(n); p.sk != nil && (len(p.sk.stages) == 2 || len(p.sk.stages) == 3) {
			var f []string
			for _, st := range p.sk.stages {
				f = append(f, fmt.Sprintf("%d/%d", st.r, st.split))
			}
			fmt.Printf("FRAME %d %s %v\n", n, strings.Join(f, "·"), p.sk.zn)
		}
	}
}

var znExport = flag.String("zn.export", "", "directory: TestZnExport writes its inputs and outputs there")

// TestZnExport writes, for every length up to 512 whose plan has two or three
// Stockham passes, a random input and go-fft's FFT and IFFT of it, and for the real
// transforms of twice that length a random input, its RFFT, and the IRFFT of
// that RFFT, as raw little-endian float64 files (complex as re, im pairs), so
// that numpy can be run on the exact bytes go-fft read
// (benchmarks/results/round32-zen3-small-20261009/scripts/zn_accuracy.py).
func TestZnExport(t *testing.T) {
	if *znExport == "" {
		t.Skip("set -zn.export=DIR")
	}
	rng := rand.New(rand.NewPCG(32, 9))
	put := func(name string, x []float64) {
		b := make([]byte, 8*len(x))
		for i, v := range x {
			binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(v))
		}
		if err := os.WriteFile(filepath.Join(*znExport, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	flat := func(z []complex128) []float64 {
		x := make([]float64, 0, 2*len(z))
		for _, v := range z {
			x = append(x, real(v), imag(v))
		}
		return x
	}
	var list []string
	for n := 2; n <= 512; n++ {
		p := NewPlan(n)
		if p.sk == nil || len(p.sk.stages) != 2 && len(p.sk.stages) != 3 {
			continue
		}
		list = append(list, fmt.Sprintf("%d %v", n, p.sk.zn != nil))
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
		put(fmt.Sprintf("c%d.in", n), flat(src))
		put(fmt.Sprintf("c%d.fft", n), flat(p.FFT(make([]complex128, n), src)))
		put(fmt.Sprintf("c%d.ifft", n), flat(p.IFFT(make([]complex128, n), src)))
		r := NewRealPlan(2 * n)
		x := make([]float64, 2*n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		spec := r.RFFT(make([]complex128, n+1), x)
		put(fmt.Sprintf("r%d.in", 2*n), x)
		put(fmt.Sprintf("r%d.rfft", 2*n), flat(spec))
		put(fmt.Sprintf("r%d.irfft", 2*n), r.IRFFT(make([]float64, 2*n), spec))
	}
	if err := os.WriteFile(filepath.Join(*znExport, "lengths.txt"), []byte(strings.Join(list, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
