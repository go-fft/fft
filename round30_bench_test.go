package fft

// The measurements behind BENCHMARKS.md's Round 30: the smallest sizes,
// complex 256 and RFFT 256, on Zen 3 and Neoverse-N1, part by part, in
// interleaved rounds whose order rotates every round.

import (
	"flag"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

var (
	smallAB    = flag.String("r30.ab", "", "rounds,ms: run Round 30's interleaved tests")
	smallReals = flag.String("r30.real", "256,512,1024,4096", "RFFT lengths TestR30Real splits")
)

func smallRounds(t *testing.T) (rounds, ms int) {
	if *smallAB == "" {
		t.Skip("set -r30.ab=rounds,ms")
	}
	if _, err := fmt.Sscanf(*smallAB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	return rounds, ms
}

func smallMedian(x []float64) float64 {
	y := slices.Clone(x)
	slices.Sort(y)
	if len(y)%2 == 1 {
		return y[len(y)/2]
	}
	return (y[len(y)/2-1] + y[len(y)/2]) / 2
}

// smallRotate times fns in rounds, the order rotated every round, each call
// repeated iters times (calibrated on fns[0]), and returns the per-round ns.
func smallRotate(fns []func(), rounds, ms int) [][]float64 {
	iters := 1
	for {
		t0 := time.Now()
		for range iters {
			fns[0]()
		}
		if time.Since(t0) > time.Duration(ms)*time.Millisecond/4 {
			iters *= 4
			break
		}
		iters *= 2
	}
	times := make([][]float64, len(fns))
	for round := range rounds {
		for q := range fns {
			v := (q + round) % len(fns)
			t0 := time.Now()
			for range iters {
				fns[v]()
			}
			times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(iters))
		}
	}
	return times
}

// smallRatio prints the median of a/b per round and its spread.
func smallRatio(a, b []float64) string {
	r := make([]float64, len(a))
	for i := range r {
		r[i] = a[i] / b[i]
	}
	return fmt.Sprintf("%.3f (%.3f-%.3f)", smallMedian(r), slices.Min(r), slices.Max(r))
}

// TestR30Real splits RFFT and IRFFT at each -r30.real length into the
// half-length complex transform, the untangle (retangle), and the two
// sync.Pool round trips, each timed alone, rotated with the wholes.
func TestR30Real(t *testing.T) {
	rounds, ms := smallRounds(t)
	for _, s := range strings.Split(*smallReals, ",") {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		m := n / 2
		Z := make([]complex128, m)
		zin := asComplex(src[:2*m])
		p.half.execute(Z, zin, false)
		p.RFFT(dst, src)
		back := make([]float64, n)
		zb := make([]complex128, m)
		names := []string{"rfft", "half", "untangle", "realpool", "halfpool", "irfft", "retangle", "halfinv"}
		fns := []func(){
			func() { p.RFFT(dst, src) },
			func() { p.half.execute(Z, zin, false) },
			func() { rfftUntangle(dst, Z, p.tw, m) },
			func() { p.scratch.Put(p.scratch.Get()) },
			func() {
				if sk := p.half.sk; sk != nil {
					sk.scratch.Put(sk.scratch.Get())
				}
			},
			func() { p.IRFFT(back, dst) },
			func() { irfftRetangle(zb, dst, p.tw, m, 0.5/float64(m)) },
			func() { p.half.execute(Z, zb, true) },
		}
		times := smallRotate(fns, rounds, ms)
		var b strings.Builder
		for i, nm := range names {
			fmt.Fprintf(&b, " %s %.1f", nm, smallMedian(times[i]))
		}
		var f []int
		if p.half.sk != nil {
			for _, st := range p.half.sk.stages {
				f = append(f, st.r)
			}
		}
		fmt.Printf("REAL %d half(%s)%s\n", n, factorName(f), b.String())
	}
}

// BenchmarkR30AB is the row set Round 30's A/B runs between binaries time:
// the touched rows (RFFT and IRFFT, complex 128 and 256) first, then
// untouched ones in the same process.
func BenchmarkR30AB(b *testing.B) {
	for _, n := range []int{256, 512, 1024, 4096} {
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		back := make([]float64, n)
		p.RFFT(dst, src)
		b.Run("R"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, src)
			}
		})
		b.Run("I"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IRFFT(back, dst)
			}
		})
	}
	for _, n := range []int{128, 256, 512, 1024, 1000, 4096} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
}

// TestR30Direct times RFFT and IRFFT with the RealPlan's pooled buffer
// (the pool) and without it (direct: the half transform writes dst and the
// untangle runs in place), rotated.
func TestR30Direct(t *testing.T) {
	rounds, ms := smallRounds(t)
	defer func(v bool) { smallRealDirect = v }(smallRealDirect)
	for _, s := range strings.Split(*smallReals, ",") {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		p.RFFT(dst, src)
		spec := slices.Clone(dst)
		back := make([]float64, n)
		times := smallRotate([]func(){
			func() { smallRealDirect = false; p.RFFT(dst, src) },
			func() { smallRealDirect = true; p.RFFT(dst, src) },
			func() { smallRealDirect = false; p.IRFFT(back, spec) },
			func() { smallRealDirect = true; p.IRFFT(back, spec) },
		}, rounds, ms)
		fmt.Printf("DIRECT %d rfft pool %.1f direct %.1f [pool/direct %s] irfft pool %.1f direct %.1f [pool/direct %s]\n", n,
			smallMedian(times[0]), smallMedian(times[1]), smallRatio(times[0], times[1]),
			smallMedian(times[2]), smallMedian(times[3]), smallRatio(times[2], times[3]))
	}
}

// TestR30DirectSame checks the direct path against the pooled one bit for
// bit (exploration).
func TestR30DirectSame(t *testing.T) {
	defer func(v bool) { smallRealDirect = v }(smallRealDirect)
	for _, n := range []int{2, 4, 6, 8, 10, 16, 30, 64, 100, 128, 256, 512, 1000, 1024, 2048, 4096, 2 * 1009, 2 * 1201} {
		p := NewRealPlan(n)
		src := benchReal(n)
		smallRealDirect = false
		a := p.RFFT(make([]complex128, n/2+1), src)
		ia := p.IRFFT(make([]float64, n), a)
		ib2 := p.IRFFT(make([]float64, n), a[:n/4+1])
		smallRealDirect = true
		b := p.RFFT(make([]complex128, n/2+1), src)
		ib := p.IRFFT(make([]float64, n), a)
		ib3 := p.IRFFT(make([]float64, n), a[:n/4+1])
		for k := range a {
			if !sameBits(a[k], b[k]) {
				t.Fatalf("n=%d bin %d", n, k)
			}
		}
		for k := range ia {
			if math.Float64bits(ia[k]) != math.Float64bits(ib[k]) || math.Float64bits(ib2[k]) != math.Float64bits(ib3[k]) {
				t.Fatalf("n=%d sample %d", n, k)
			}
		}
	}
}

var smallSets = flag.String("r30.sets", "", `factorizations to compare, "128:4x4x8,8x4x4;256:..." (first is the reference)`)

// TestR30Orders times each listed factorization's whole transform and each
// of its passes alone, all rotated, the first factorization the reference.
func TestR30Orders(t *testing.T) {
	rounds, ms := smallRounds(t)
	for _, spec := range strings.Split(*smallSets, ";") {
		ns, vs, ok := strings.Cut(spec, ":")
		if !ok {
			continue
		}
		n, err := strconv.Atoi(ns)
		if err != nil {
			t.Fatal(err)
		}
		var names []string
		var fns []func()
		var npass []int
		for _, v := range strings.Split(vs, ",") {
			var f []int
			for _, x := range strings.Split(v, "x") {
				r, err := strconv.Atoi(x)
				if err != nil {
					t.Fatal(err)
				}
				f = append(f, r)
			}
			p := newSKPlanFactors(n, f)
			src := benchComplex(n)
			dst := make([]complex128, n)
			scr := make([]complex128, n)
			fns = append(fns, func() { p.transform(dst, src, false) })
			names = append(names, v)
			s := len(p.stages)
			in := src
			for k := range p.stages {
				out := scr
				if (s-1-k)%2 == 0 {
					out = dst
				}
				st, i0, o0 := &p.stages[k], in, out
				fns = append(fns, func() { st.pass(o0, i0, false) })
				names = append(names, fmt.Sprintf("  r%d/i%d/s%d", st.r, st.ido, st.split))
				in = out
			}
			npass = append(npass, s)
		}
		times := smallRotate(fns, rounds, ms)
		ref := times[0]
		for i, nm := range names {
			if strings.HasPrefix(nm, " ") {
				fmt.Printf("ORDER %d %s %.1f\n", n, nm, smallMedian(times[i]))
				continue
			}
			fmt.Printf("ORDER %d %-12s %.1f [ref/this %s]\n", n, nm, smallMedian(times[i]), smallRatio(ref, times[i]))
		}
		_ = npass
	}
}
