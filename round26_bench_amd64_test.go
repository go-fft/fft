//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 26: composites and small real
// sizes on amd64 (AMD Zen 3), pass by pass and factorization against
// factorization, in interleaved rounds whose order rotates every round.

import (
	"flag"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

var (
	comp2AB    = flag.String("r26.ab", "", "rounds,ms: run TestR26AB / TestR26Passes / TestR26Real")
	comp2Sets  = flag.String("r26.sets", "", `factorizations to compare, "n:rule,5x5x5x8;1296:rule,3x3x12x12"`)
	comp2Reals = flag.String("r26.real", "256,512,1000,1024,1080,1920,2000,4096", "RFFT lengths TestR26Real splits")
	comp2Max   = flag.Int("r26.max", 1<<14, "largest length TestR26Rule and TestR26Sweep time")
	comp2Min   = flag.Int("r26.min", 6, "smallest length TestR26Sweep times")
)

// comp2Variants parses -r26.sets into, per length, the named factorizations
// ("rule" is skFactorize's).
func comp2Variants(t *testing.T) (ns []int, sets map[int][][]int) {
	sets = map[int][][]int{}
	for _, spec := range strings.Split(*comp2Sets, ";") {
		ns0, vs, ok := strings.Cut(spec, ":")
		if !ok {
			continue
		}
		n, err := strconv.Atoi(ns0)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range strings.Split(vs, ",") {
			f := skFactorize(n)
			if v != "rule" {
				f = nil
				for _, x := range strings.Split(v, "x") {
					r, err := strconv.Atoi(x)
					if err != nil {
						t.Fatal(err)
					}
					f = append(f, r)
				}
			}
			if product(f) != n {
				t.Fatalf("%d: %v multiplies to %d", n, f, product(f))
			}
			sets[n] = append(sets[n], f)
		}
		ns = append(ns, n)
	}
	return ns, sets
}

func comp2Rounds(t *testing.T) (rounds, ms int) {
	if *comp2AB == "" {
		t.Skip("set -r26.ab=rounds,ms")
	}
	if _, err := fmt.Sscanf(*comp2AB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	return rounds, ms
}

// comp2Rotate times fns in rounds, the order rotated every round, each call
// repeated iters times (calibrated on fns[0] to about ms/4... ms), and returns
// the per-round ns of each.
func comp2Rotate(fns []func(), rounds, ms int) [][]float64 {
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

// TestR26AB times each length's factorizations against the first.
func TestR26AB(t *testing.T) {
	rounds, ms := comp2Rounds(t)
	ns, sets := comp2Variants(t)
	for _, n := range ns {
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		for _, f := range sets[n] {
			p := newSKPlanFactors(n, f)
			fns = append(fns, func() { p.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		for v, f := range sets[n] {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			fmt.Printf("AB %d %-18s %10.1f ns  first/this %.3f  spread %.3f\n", n, factorName(f),
				median(times[v]), median(ratios), slices.Max(ratios)/slices.Min(ratios))
		}
	}
}

// TestR26Passes times every pass of each listed factorization alone, on the
// buffers and in the layout the transform gives it, the passes and the whole
// transform rotated.
func TestR26Passes(t *testing.T) {
	rounds, ms := comp2Rounds(t)
	ns, sets := comp2Variants(t)
	for _, n := range ns {
		for _, f := range sets[n] {
			p := newSKPlanFactors(n, f)
			src := benchComplex(n)
			dst := make([]complex128, n)
			scr := make([]complex128, n)
			fns := []func(){func() { p.transform(dst, src, false) }}
			s := len(p.stages)
			in := src
			for k := range p.stages {
				out := scr
				if (s-1-k)%2 == 0 {
					out = dst
				}
				st, i0, o0 := &p.stages[k], in, out
				fns = append(fns, func() { st.pass(o0, i0, false) })
				in = out
			}
			times := comp2Rotate(fns, rounds, ms)
			var b strings.Builder
			sum := 0.0
			for k := range p.stages {
				st := &p.stages[k]
				m := median(times[k+1])
				sum += m
				fmt.Fprintf(&b, " r%d/i%d/s%d %.0f", st.r, st.ido, st.split, m)
			}
			fmt.Printf("PASSES %d %-18s whole %.0f sum %.0f |%s\n", n, factorName(f), median(times[0]), sum, b.String())
		}
	}
}

// TestR26Real splits RFFT at each -r26.real length into its half-length
// complex transform and its untangle, each timed alone, rotated with the
// whole.
func TestR26Real(t *testing.T) {
	rounds, ms := comp2Rounds(t)
	for _, s := range strings.Split(*comp2Reals, ",") {
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
		fns := []func(){
			func() { p.RFFT(dst, src) },
			func() { p.half.execute(Z, zin, false) },
			func() { rfftUntangle(dst, Z, p.tw, m) },
			func() { p.half.execute(Z, zin, false); rfftUntangle(dst, Z, p.tw, m) },
		}
		times := comp2Rotate(fns, rounds, ms)
		var f []int
		if p.half.sk != nil {
			f = p.half.sk.factors()
		}
		fmt.Printf("REAL %d whole %.0f half(%s) %.0f untangle %.0f half+untangle %.0f\n", n, median(times[0]), factorName(f), median(times[1]), median(times[2]), median(times[3]))
	}
}

// comp2IntelRule is Round 24's Intel composite rule (radix-12 passes, else
// a radix-16 tail) for n, nil where it keeps skFactorizeOrder's.
func comp2IntelRule(n int) []int {
	if f := compRadix12For(n, true); f != nil {
		return f
	}
	return compRadix16For(n, true)
}

// TestR26Rule times, for every n = 2^e·3^a·5^b up to -r26.max with a+b >= 1
// that Round 24's Intel rule changes, the current factorization against the
// Intel one, in interleaved rounds.
func TestR26Rule(t *testing.T) {
	rounds, ms := comp2Rounds(t)
	for n := 6; n <= *comp2Max; n++ {
		g := comp2IntelRule(n)
		if g == nil {
			continue
		}
		f := skFactorize(n)
		if slices.Equal(f, g) {
			continue
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		pf, pg := newSKPlanFactors(n, f), newSKPlanFactors(n, g)
		times := comp2Rotate([]func(){func() { pf.transform(dst, src, false) }, func() { pg.transform(dst, src, false) }}, rounds, ms)
		ratios := make([]float64, rounds)
		for i := range ratios {
			ratios[i] = times[0][i] / times[1][i]
		}
		fmt.Printf("RULE %d %s %.1f %s %.1f cur/intel %.3f spread %.3f\n", n, factorName(f), median(times[0]),
			factorName(g), median(times[1]), median(ratios), slices.Max(ratios)/slices.Min(ratios))
	}
}

// comp2Tails are the power-of-two tails tried for 2^e: radix8Maximal's (radix
// 4 before 8) and, where Round 24 found one, the radix-16 tail.
func comp2Tails(e int) [][]int {
	t := radix8Maximal(e)
	slices.Reverse(t)
	out := [][]int{t}
	switch e {
	case 4:
		out = append(out, []int{16})
	case 7:
		out = append(out, []int{8, 16})
	case 8:
		out = append(out, []int{16, 16})
	case 10:
		out = append(out, []int{8, 8, 16})
	}
	return out
}

// comp2Candidates returns the factorizations TestR26Sweep times for n =
// 2^e·3^a·5^b: t20 radix-20, t12 radix-12, t10 (0 or 1) radix-10 and t15 (0
// or 1) radix-15 passes, the remaining 3s and 5s first, then (with the 15
// either first or after them) the power-of-two tail, then the 10, 12 and 20
// passes. The current factorization comes first.
func comp2Candidates(n int) [][]int {
	e, odd := 0, n
	for odd%2 == 0 {
		odd /= 2
		e++
	}
	a, b := 0, 0
	for ; odd%3 == 0; odd /= 3 {
		a++
	}
	for ; odd%5 == 0; odd /= 5 {
		b++
	}
	if odd != 1 {
		return nil
	}
	out := [][]int{skFactorize(n)}
	seen := map[string]bool{factorName(out[0]): true}
	rep := func(f []int, r, c int) []int {
		for range c {
			f = append(f, r)
		}
		return f
	}
	for t20 := 0; t20 <= min(b, e/2); t20++ {
		for t12 := 0; t12 <= min(a, (e-2*t20)/2); t12++ {
			for t10 := 0; t10 <= min(1, b-t20, e-2*t20-2*t12); t10++ {
				for t15 := 0; t15 <= min(1, a-t12, b-t20-t10); t15++ {
					ep := e - 2*t20 - 2*t12 - t10
					for _, tail := range comp2Tails(ep) {
						for _, late := range []bool{false, true} {
							if late && t15 == 0 {
								continue
							}
							var f []int
							f = rep(f, 3, a-t12-t15)
							f = rep(f, 5, b-t20-t10-t15)
							if !late {
								f = rep(f, 15, t15)
							}
							f = append(f, tail...)
							if late {
								f = rep(f, 15, t15)
							}
							f = rep(f, 10, t10)
							f = rep(f, 12, t12)
							f = rep(f, 20, t20)
							if product(f) != n {
								panic(fmt.Sprint(n, f))
							}
							if k := factorName(f); !seen[k] {
								seen[k] = true
								out = append(out, f)
							}
						}
					}
				}
			}
		}
	}
	return out
}

// TestR26Sweep times comp2Candidates for every n = 2^e·3^a·5^b (a+b >= 1, e
// >= 1) from -r26.min to -r26.max, in interleaved rounds.
func TestR26Sweep(t *testing.T) {
	rounds, ms := comp2Rounds(t)
	for n := *comp2Min; n <= *comp2Max; n++ {
		if n%2 != 0 || n&(n-1) == 0 {
			continue
		}
		cs := comp2Candidates(n)
		if len(cs) < 2 {
			continue
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		for _, f := range cs {
			p := newSKPlanFactors(n, f)
			fns = append(fns, func() { p.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		var b strings.Builder
		for v, f := range cs {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			fmt.Fprintf(&b, " %s=%.0f/%.3f", factorName(f), median(times[v]), median(ratios))
		}
		fmt.Printf("SWEEP %d%s\n", n, b.String())
	}
}

// TestR26Overhead splits a small transform's time into its passes and what
// surrounds them: Plan.FFT, skPlan.transform, the passes on a fixed scratch
// buffer, and the scratch pool's Get and Put alone.
func TestR26Overhead(t *testing.T) {
	rounds, ms := comp2Rounds(t)
	for _, n := range []int{64, 128, 256, 512, 1024, 1000} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		sk := p.sk
		scr := make([]complex128, 2*n)[:n]
		passes := func() {
			s := len(sk.stages)
			in := src
			for k := range sk.stages {
				out := scr
				if (s-1-k)%2 == 0 {
					out = dst
				}
				sk.stages[k].pass(out, in, false)
				in = out
			}
		}
		times := comp2Rotate([]func(){
			func() { p.FFT(dst, src) },
			func() { sk.transform(dst, src, false) },
			passes,
			func() { sk.scratch.Put(sk.scratch.Get()) },
		}, rounds, ms)
		fmt.Printf("OVER %d %s FFT %.1f transform %.1f passes %.1f pool %.1f\n", n, factorName(sk.factors()),
			median(times[0]), median(times[1]), median(times[2]), median(times[3]))
	}
}

// BenchmarkR26AB is the row set Round 26's interleaved A/B runs time: the
// composites, powers of two, the prime engines (Rader 1009 and 1201 have
// 2^4·3^2·7 and 2^4·3·5^2 convolutions, Bluestein 10007 a 20480-point one),
// lengths with a factor 7, real transforms (whose halves are 128 … 2048,
// 500, 540 and 960 points) and 2-D plans of composite sides.
func BenchmarkR26AB(b *testing.B) {
	compR24Warm(b)
	for _, n := range []int{1000, 1080, 1296, 1920, 2000, 6000, 240, 480, 720, 960, 1200, 1440, 3600, 7200, 15360, 16000,
		128, 256, 512, 1024, 4096, 65536, 1009, 1201, 10007, 1008, 2100} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
	for _, n := range []int{256, 512, 1000, 1024, 1080, 1920, 2000, 4096} {
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		b.Run("R"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, src)
			}
		})
	}
	for _, n := range []int{100, 120, 240} {
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
