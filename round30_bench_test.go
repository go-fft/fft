package fft

// The measurements behind BENCHMARKS.md's Round 30: the smallest sizes,
// complex 256 and RFFT 256, on Zen 3 and Neoverse-N1, part by part, in
// interleaved rounds whose order rotates every round.

import (
	"flag"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"
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
// the touched rows (RFFT and IRFFT; on arm64 the complex powers of two from
// 64 to 4096 and the 2-D plans built on them) first, then untouched ones in
// the same process.
func BenchmarkR30AB(b *testing.B) {
	for _, n := range []int{256, 512, 1024, 2048, 4096, 8192, 1000} {
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
	for _, n := range []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 1000, 1080} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
	for _, n := range []int{64, 128, 256} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		b.Run("D"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
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

var smallSweepE = flag.String("r30.sweep", "5,13", "lo,hi: the powers of two TestR30Pow2Sweep times")

// smallPow2Orders lists every ordered factorization of 2^e into radices 2, 4
// and 8 with at most one radix 2.
func smallPow2Orders(e int) [][]int {
	var out [][]int
	var rec func(rest int, cur []int, twos int)
	rec = func(rest int, cur []int, twos int) {
		if rest == 0 {
			out = append(out, slices.Clone(cur))
			return
		}
		for _, b := range []int{1, 2, 3} {
			if b > rest || (b == 1 && twos > 0) {
				continue
			}
			t := twos
			if b == 1 {
				t++
			}
			rec(rest-b, append(cur, 1<<b), t)
		}
	}
	rec(e, nil, 0)
	return out
}

// TestR30Pow2Sweep times, for each 2^e, skFactorize's factorization against
// every ordered factorization into radices 2, 4 and 8, rotated, and prints
// them fastest first.
func TestR30Pow2Sweep(t *testing.T) {
	rounds, ms := smallRounds(t)
	var lo, hi int
	if _, err := fmt.Sscanf(*smallSweepE, "%d,%d", &lo, &hi); err != nil {
		t.Fatal(err)
	}
	for e := lo; e <= hi; e++ {
		n := 1 << e
		rule := skFactorize(n)
		sets := [][]int{rule}
		for _, f := range smallPow2Orders(e) {
			if !slices.Equal(f, rule) {
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
		times := smallRotate(fns, rounds, ms)
		idx := make([]int, len(sets))
		for i := range idx {
			idx[i] = i
		}
		slices.SortFunc(idx, func(a, b int) int {
			ma, mb := smallMedian(times[a]), smallMedian(times[b])
			if ma < mb {
				return -1
			}
			if ma > mb {
				return 1
			}
			return 0
		})
		for rank, i := range idx[:min(6, len(idx))] {
			fmt.Printf("SWEEP %d #%d %-14s %.1f [rule/this %s]\n", n, rank, factorName(sets[i]), smallMedian(times[i]), smallRatio(times[0], times[i]))
		}
		fmt.Printf("SWEEP %d rule %-14s %.1f\n", n, factorName(rule), smallMedian(times[0]))
	}
}

// TestR30Offsets times one n-point transform (-r30.n) with src, dst and the
// scratch buffer placed at chosen offsets modulo 4 KB in one arena: whether
// an untouched row's time depends on where the heap put its buffers.
func TestR30Offsets(t *testing.T) {
	rounds, ms := smallRounds(t)
	n := *smallOffN
	p := NewPlan(n)
	if p.sk == nil {
		t.Skip("not a Stockham length")
	}
	arena := make([]complex128, 4*4096/16+8*n+4096)
	base := int((4096 - uintptr(unsafe.Pointer(&arena[0]))%4096) % 4096 / 16)
	at := func(page, off int) []complex128 { // off in bytes, multiple of 16
		i := base + page*(4096/16)*((n*16+4095)/4096+1) + off/16
		return arena[i : i+n : i+n]
	}
	src := at(0, 0)
	copy(src, benchComplex(n))
	var names []string
	var fns []func()
	for _, d := range []int{0, 256, 512, 1024, 2048, 3072} {
		for _, s := range []int{0, 576, 1024, 2048} {
			dst, scr := at(1, d), at(2, s)
			names = append(names, fmt.Sprintf("dst+%d scr+%d", d, s))
			fns = append(fns, func() { p.sk.run(dst, src, scr, false) })
		}
	}
	times := smallRotate(fns, rounds, ms)
	for i, nm := range names {
		fmt.Printf("OFF %d %-18s %.1f\n", n, nm, smallMedian(times[i]))
	}
}

var smallOffN = flag.Int("r30.n", 64, "TestR30Offsets' length")
