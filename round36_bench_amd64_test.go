//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 36: the rows where FFTW still
// leads on Zen 3, located part by part, in interleaved rounds whose order
// rotates every round (comp2Rotate).

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

var (
	zmLoc   = flag.String("r36.loc", "", "rounds,ms: run TestZmLocate / TestZmConv")
	zmBlue  = flag.String("r36.blue", "10007", "Bluestein lengths TestZmLocate and TestZmConv take apart")
	zmConvW = flag.Float64("r36.convw", 1.4, "TestZmConv's window: 7-smooth m in [2n-1, w·(2n-1)]")
)

func zmRounds(t *testing.T) (rounds, ms int) {
	if *zmLoc == "" {
		t.Skip("set -r36.loc=rounds,ms")
	}
	if _, err := fmt.Sscanf(*zmLoc, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	return rounds, ms
}

func zmInts(t *testing.T, s string) []int {
	var ns []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(f)
		if err != nil {
			t.Fatal(err)
		}
		ns = append(ns, n)
	}
	return ns
}

// zmBluesteinM is newBluesteinPlan with the convolution length m given.
func zmBluesteinM(n, m int) *bluesteinPlan {
	p := &bluesteinPlan{n: n, m: m}
	p.wF = make([]complex128, n)
	p.wI = make([]complex128, n)
	for j := 0; j < n; j++ {
		jj := (j * j) % (2 * n)
		ang := math.Pi * float64(jj) / float64(n)
		p.wF[j] = complex(math.Cos(-ang), math.Sin(-ang))
		p.wI[j] = complex(math.Cos(ang), math.Sin(ang))
	}
	p.bF = buildKernelSpectrum(p.wF, n, m)
	p.bI = buildKernelSpectrum(p.wI, n, m)
	p.scratch.New = func() any { b := make([]complex128, m); return &b }
	return p
}

// TestZmLocate takes Bluestein and the 2-D 64² transform apart.
func TestZmLocate(t *testing.T) {
	rounds, ms := zmRounds(t)
	for _, n := range zmInts(t, *zmBlue) {
		p := NewPlan(n)
		bp := p.bluestein
		if bp == nil {
			t.Fatalf("%d is not Bluestein", n)
		}
		m := bp.m
		src := benchComplex(n)
		dst := make([]complex128, n)
		a := make([]complex128, m)
		cp := cachedPlan(m)
		w := bp.wF
		fns := []func(){
			func() { p.FFT(dst, src) },
			func() {
				for j := 0; j < n; j++ {
					a[j] = src[j] * w[j]
				}
				clear(a[n:])
			},
			func() { cp.execute(a, a, false) },
			func() { kernels.CMul(a, bp.bF) },
			func() { cp.execute(a, a, true) },
			func() {
				for k := 0; k < n; k++ {
					dst[k] = a[k] * w[k]
				}
			},
		}
		times := comp2Rotate(fns, rounds, ms)
		var f []int
		if cp.sk != nil {
			f = cp.sk.factors()
		}
		fmt.Printf("BLUE %d m %d (%s) whole %.0f chirp-in %.0f fft %.0f cmul %.0f ifft %.0f chirp-out %.0f\n", n, m, factorName(f),
			median(times[0]), median(times[1]), median(times[2]), median(times[3]), median(times[4]), median(times[5]))
	}
	for _, n := range []int{64, 128} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		fns := []func(){
			func() { p.FFT(dst, src) },
			func() { p.transformAxis(dst, src, 0, false) },
			func() { p.transformAxis(dst, dst, 1, false) },
			func() { NewPlan(n).sk.transform(dst[:n], src[:n], false) },
		}
		pl := cachedPlan(n)
		fns[3] = func() { pl.execute(dst[:n], src[:n], false) }
		times := comp2Rotate(fns, rounds, ms)
		fmt.Printf("FFT2 %dx%d whole %.0f axis0(strips %v) %.0f axis1(rows) %.0f one-row-plan %.0f (x%d = %.0f)\n", n, n,
			median(times[0]), p.strips[0] != nil, median(times[1]), median(times[2]), median(times[3]), n, float64(n)*median(times[3]))
	}
}

// TestZmConv times Bluestein at every 7-smooth convolution length in the
// window, against the one bestConvLen picks.
func TestZmConv(t *testing.T) {
	rounds, ms := zmRounds(t)
	for _, n := range zmInts(t, *zmBlue) {
		lo := 2*n - 1
		hi := int(float64(lo) * *zmConvW)
		cur := bestConvLen(lo)
		ms0 := []int{cur}
		for m := lo; m <= hi; m++ {
			if _, ok := convCost(m); ok && m != cur {
				ms0 = append(ms0, m)
			}
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		for _, m := range ms0 {
			bp := zmBluesteinM(n, m)
			fns = append(fns, func() { bp.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		for v, m := range ms0 {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			var f []int
			if sk := cachedPlan(m).sk; sk != nil {
				f = sk.factors()
			}
			fmt.Printf("CONV %d m %d %-22s %10.0f ns  cur/this %.3f\n", n, m, factorName(f), median(times[v]), median(ratios))
		}
	}
}

// zmBlueOld is bluesteinPlan.transform before Round 36: the pointwise
// products as Go loops and kernels.CMul.
func zmBlueOld(p *bluesteinPlan, dst, src []complex128) {
	n, w, bSpec := p.n, p.wF, p.bF
	bp := p.scratch.Get().(*[]complex128)
	defer p.scratch.Put(bp)
	a := *bp
	for j := 0; j < n; j++ {
		a[j] = src[j] * w[j]
	}
	clear(a[n:])
	plan := cachedPlan(p.m)
	plan.execute(a, a, false)
	kernels.CMul(a, bSpec)
	plan.execute(a, a, true)
	for k := 0; k < n; k++ {
		dst[k] = a[k] * w[k]
	}
}

// zmRowsOld is PlanN.contiguousLines before Round 36: every Stockham line
// through run on one pooled buffer.
func zmRowsOld(p *PlanN, dst, src []complex128, ax, lo, hi int) {
	n, sk := p.shape[ax], p.axes[ax].sk
	bp := sk.scratch.Get().(*[]complex128)
	for c := lo; c < hi; c++ {
		sk.run(dst[c*n:(c+1)*n], src[c*n:(c+1)*n], *bp, false)
	}
	sk.scratch.Put(bp)
}

var zmABRows = flag.String("r36.rows", "blue:10007,blue:4099,blue:30011,2d:64,2d:128,2d:256,2d:32,c:1024,c:1000,r:4096,c:64",
	"TestZmAB's rows: blue:n (old products ÷ CMulTo), 2d:n (old rows ÷ frame rows), c:n / r:n (untouched, timed twice)")

// TestZmAB times each touched row against its pre-Round-36 code in this
// process, and the untouched rows against themselves (old ÷ new should
// read 1).
func TestZmAB(t *testing.T) {
	rounds, ms := zmRounds(t)
	for _, spec := range strings.Split(*zmABRows, ",") {
		kind, ns, _ := strings.Cut(spec, ":")
		n, err := strconv.Atoi(ns)
		if err != nil {
			t.Fatal(err)
		}
		var oldF, newF func()
		switch kind {
		case "blue":
			p := zmBluesteinM(n, bestConvLen(2*n-1))
			src := benchComplex(n)
			dst := make([]complex128, n)
			oldF = func() { zmBlueOld(p, dst, src) }
			newF = func() { p.transform(dst, src, false) }
		case "2d":
			p := NewPlanN(n, n)
			src := benchComplex(n * n)
			dst := make([]complex128, n*n)
			oldF = func() { p.transformAxis(dst, src, 0, false); zmRowsOld(p, dst, dst, 1, 0, n) }
			newF = func() { p.FFT(dst, src) }
		case "c":
			p := NewPlan(n)
			src := benchComplex(n)
			dst := make([]complex128, n)
			oldF = func() { p.FFT(dst, src) }
			newF = oldF
		case "r":
			p := NewRealPlan(n)
			src := benchReal(n)
			dst := make([]complex128, n/2+1)
			oldF = func() { p.RFFT(dst, src) }
			newF = oldF
		default:
			t.Fatalf("row %q", spec)
		}
		times := comp2Rotate([]func(){oldF, newF}, rounds, ms)
		ratios := make([]float64, rounds)
		for i := range ratios {
			ratios[i] = times[0][i] / times[1][i]
		}
		fmt.Printf("ZMAB %-11s old %10.0f new %10.0f  old/new %.3f  range %.3f–%.3f\n", spec,
			median(times[0]), median(times[1]), median(ratios), slices.Min(ratios), slices.Max(ratios))
	}
}

var zmExport = flag.String("r36.export", "", "directory TestZmExport writes inputs and outputs to")

// TestZmExport writes, for numpy to check (scripts/zm_accuracy.py), random
// inputs and go-fft's outputs as raw little-endian float64: Bluestein's FFT
// and IFFT, and FFT2 and IFFT2 of n×n plans whose rows run on the frame.
func TestZmExport(t *testing.T) {
	if *zmExport == "" {
		t.Skip("set -r36.export=dir")
	}
	r := rand.New(rand.NewPCG(36, 2026))
	sig := func(n int) []complex128 {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		return x
	}
	put := func(name string, x []complex128) {
		b := make([]byte, 16*len(x))
		for i, v := range x {
			binary.LittleEndian.PutUint64(b[16*i:], math.Float64bits(real(v)))
			binary.LittleEndian.PutUint64(b[16*i+8:], math.Float64bits(imag(v)))
		}
		if err := os.WriteFile(filepath.Join(*zmExport, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var list strings.Builder
	for _, n := range []int{23, 47, 59, 94, 4099, 10007, 30011} {
		p := NewPlan(n)
		if p.bluestein == nil {
			t.Fatalf("%d is not Bluestein", n)
		}
		x := sig(n)
		put(fmt.Sprintf("b%d.in", n), x)
		put(fmt.Sprintf("b%d.fft", n), p.FFT(make([]complex128, n), x))
		put(fmt.Sprintf("b%d.ifft", n), p.IFFT(make([]complex128, n), x))
		fmt.Fprintf(&list, "b %d %d\n", n, p.bluestein.m)
	}
	for _, n := range []int{2, 3, 8, 12, 16, 20, 32, 48, 60, 64, 100, 128, 200, 256} {
		p := NewPlanN(n, n)
		x := sig(n * n)
		put(fmt.Sprintf("d%d.in", n), x)
		put(fmt.Sprintf("d%d.fft", n), p.FFT(make([]complex128, n*n), x))
		put(fmt.Sprintf("d%d.ifft", n), p.IFFT(make([]complex128, n*n), x))
		fmt.Fprintf(&list, "d %d %v\n", n, p.axes[1].sk != nil && p.axes[1].sk.zn != nil)
	}
	if err := os.WriteFile(filepath.Join(*zmExport, "list.txt"), []byte(list.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

var zmTailsN = flag.String("r36.tails", "6144,10240,12288,20480", "TestZmTails' lengths")

// zmPow2Tails lists every ordered sequence of radices 4, 8 and 16 whose
// product is 2^e, at most six passes.
func zmPow2Tails(e int) [][]int {
	var out [][]int
	var rec func(left int, cur []int)
	rec = func(left int, cur []int) {
		if left == 0 {
			out = append(out, slices.Clone(cur))
			return
		}
		if len(cur) == 6 {
			return
		}
		for _, b := range []int{2, 3, 4} {
			if b <= left {
				rec(left-b, append(cur, 1<<b))
			}
		}
	}
	rec(e, nil)
	return out
}

// TestZmTails times, for each length 2^e·odd, the current factorization
// against its odd passes followed by every power-of-two tail of radices 4,
// 8 and 16 (zmPow2Tails), rotated.
func TestZmTails(t *testing.T) {
	rounds, ms := zmRounds(t)
	for _, n := range zmInts(t, *zmTailsN) {
		rule := skFactorize(n)
		var odd []int
		for _, r := range rule {
			if r&(r-1) != 0 {
				odd = append(odd, r)
			}
		}
		e := 0
		for m := n / product(odd); m > 1; m /= 2 {
			e++
		}
		sets := [][]int{rule}
		for _, tail := range zmPow2Tails(e) {
			if f := slices.Concat(odd, tail); !slices.Equal(f, rule) {
				sets = append(sets, f)
			}
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		for _, f := range sets {
			p := newSKPlanFactors(n, f)
			fns = append(fns, func() { p.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		for v, f := range sets {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			fmt.Printf("TAILS %d %-22s %10.0f ns  rule/this %.3f  range %.3f–%.3f\n", n, factorName(f), median(times[v]), median(ratios), slices.Min(ratios), slices.Max(ratios))
		}
	}
}

var zmPow2N = flag.String("r36.pow2", "", "TestZmPow2Rule's lengths (default: 2^e·odd, e = 11..14, odd in 3, 5, 9, 15, 25, 27, 45, at most 2^18)")

// zmOddPart is the odd passes of skFactorize(n), in its order, and the
// power of two left.
func zmOddPart(n int) (odd []int, e int) {
	for _, r := range skFactorize(n) {
		if r&(r-1) != 0 {
			odd = append(odd, r)
		}
	}
	for m := n / product(odd); m > 1; m /= 2 {
		e++
	}
	return odd, e
}

// zmOneEight is 2^e as radix-4 passes, then (e odd) one radix-8 pass, then
// a final radix-4 pass.
func zmOneEight(e int) []int {
	var f []int
	k := e - 2
	if k%2 == 1 {
		k -= 3
	}
	for ; k > 0; k -= 2 {
		f = append(f, 4)
	}
	if (e-2)%2 == 1 {
		f = append(f, 8)
	}
	return append(f, 4)
}

// TestZmPow2Rule times, for composites 2^e·odd with e >= 11, the current
// factorization against the odd passes followed by splitPow2Factors(e) and
// by zmOneEight(e).
func TestZmPow2Rule(t *testing.T) {
	rounds, ms := zmRounds(t)
	var ns []int
	if *zmPow2N != "" {
		ns = zmInts(t, *zmPow2N)
	} else {
		for e := 11; e <= 14; e++ {
			for _, o := range []int{3, 5, 9, 15, 25, 27, 45} {
				if n := o << e; n <= 1<<18 {
					ns = append(ns, n)
				}
			}
		}
	}
	for _, n := range ns {
		odd, e := zmOddPart(n)
		sets := [][]int{skFactorize(n)}
		for _, tail := range [][]int{splitPow2Factors(e), zmOneEight(e)} {
			if f := slices.Concat(odd, tail); !slices.ContainsFunc(sets, func(g []int) bool { return slices.Equal(f, g) }) {
				sets = append(sets, f)
			}
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		for _, f := range sets {
			p := newSKPlanFactors(n, f)
			fns = append(fns, func() { p.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		for v, f := range sets {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			fmt.Printf("POW2 %d e%d %-22s %10.0f ns  rule/this %.3f  range %.3f–%.3f\n", n, e, factorName(f), median(times[v]), median(ratios), slices.Min(ratios), slices.Max(ratios))
		}
	}
}
