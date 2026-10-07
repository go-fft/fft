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
