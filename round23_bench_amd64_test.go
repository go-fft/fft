//go:build amd64

package fft

// The benchmarks behind BENCHMARKS.md's Round 23: the split layout on amd64
// against the current factorization of each power of two.

import (
	"flag"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-fft/fft/internal/kernels"
)

// splitCandidates returns the factorizations of 2^e timed with the split
// layout: every ordering of radix-4 and radix-8 passes closed by a final pass
// of radix 4, 8 or 16, with at most one radix-4 pass beyond the third radix-8
// one to keep large sizes tractable (a radix-4 pass and two radix-8 passes
// are the same bits as one more radix-8 pass and a radix 4 elsewhere).
func splitCandidates(e int) [][]int {
	var out [][]int
	for _, last := range []int{4, 8, 16} {
		lb := map[int]int{4: 2, 8: 3, 16: 4}[last]
		rest := e - lb
		for n8 := 0; 3*n8 <= rest; n8++ {
			if (rest-3*n8)%2 != 0 {
				continue
			}
			n4 := (rest - 3*n8) / 2
			if n4+n8 == 0 || (n8 > 0 && n4 > 2 && e > 14) {
				continue
			}
			var f []int
			for range n4 {
				f = append(f, 4)
			}
			for range n8 {
				f = append(f, 8)
			}
			ps := [][]int{f}
			if e <= 14 {
				ps = perms(f)
			}
			for _, q := range ps {
				out = append(out, append(append([]int(nil), q...), last))
			}
		}
	}
	return out
}

// BenchmarkSplitRules times, for every power of two from 16 to 2^20, the
// current factorization interleaved ("cur") and each candidate split ("s-")
// and, with -split.twins, interleaved ("i-").
func BenchmarkSplitRules(b *testing.B) {
	defer splitBench(false)()
	for e := splitMinE; e <= splitMaxE; e++ {
		n := 1 << e
		src := benchComplex(n)
		dst := make([]complex128, n)
		kernels.UseStockhamSplit = false
		cur := newSKPlanFactors(n, skFactorize(n))
		b.Run(strconv.Itoa(n)+"/cur-"+factorName(skFactorize(n)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				cur.transform(dst, src, false)
			}
		})
		for _, f := range splitCandidates(e) {
			for _, on := range []bool{true, false} {
				if !on && !*splitTwins {
					continue
				}
				kernels.UseStockhamSplit = on
				p := newSKPlanFactors(n, f)
				pre := map[bool]string{true: "/s-", false: "/i-"}[on]
				b.Run(strconv.Itoa(n)+pre+factorName(f), func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						p.transform(dst, src, false)
					}
				})
			}
		}
	}
}

// splitBench sets UseStockhamSplit until the returned function restores it.
func splitBench(on bool) func() {
	old := kernels.UseStockhamSplit
	kernels.UseStockhamSplit = on
	return func() { kernels.UseStockhamSplit = old }
}

var (
	splitTwins = flag.Bool("split.twins", false, "also time each split candidate interleaved")
	splitMinE  = 4
	splitMaxE  = 20
)

func init() {
	flag.IntVar(&splitMinE, "split.mine", 4, "smallest exponent BenchmarkSplitRules times")
	flag.IntVar(&splitMaxE, "split.maxe", 20, "largest exponent BenchmarkSplitRules times")
}

// BenchmarkSplitPasses times each pass of a factorization alone, split and
// interleaved, on buffers placed as the transform places them (the scratch
// off dst's sets).
func BenchmarkSplitPasses(b *testing.B) {
	defer splitBench(false)()
	for _, f := range [][]int{{4, 4, 4, 4}, {4, 8, 8}, {8, 8, 4}, {8, 8, 8}, {4, 4, 4, 4, 4}, {4, 8, 8, 4}, {8, 8, 16}, {4, 4, 8, 8, 8}, {8, 8, 8, 8}, {4, 4, 4, 4, 4, 4, 4, 4}} {
		n := 1
		for _, r := range f {
			n *= r
		}
		for _, on := range []bool{false, true} {
			kernels.UseStockhamSplit = on
			p := newSKPlanFactors(n, f)
			dst := benchComplex(n)
			buf := make([]complex128, n+setSpan)
			scr := offTheSets(buf, dst, n)
			for k := range p.stages {
				st := &p.stages[k]
				in, out := dst, scr
				if k%2 == 1 {
					in, out = scr, dst
				}
				name := strconv.Itoa(n) + "/" + factorName(f) + "/" + map[bool]string{true: "s", false: "i"}[on] +
					"/p" + strconv.Itoa(k) + "-r" + strconv.Itoa(st.r) + "-ido" + strconv.Itoa(st.ido) + "-m" + strconv.Itoa(int(st.split))
				b.Run(name, func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						st.pass(out, in, false)
					}
				})
			}
		}
	}
}

var splitAB = flag.String("split.ab", "", "TestSplitAB: rounds,ms (e.g. 7,40) to time the factorizations in splitABSets")

// splitABSets lists, per length, the factorizations TestSplitAB compares:
// "i:" interleaved, "s:" split; "cur" is skFactorize's without splitTable
// (main's), "rule" with it.
var splitABSets = map[int][]string{
	32:      {"i:cur", "s:cur"},
	64:      {"i:cur", "s:cur"},
	128:     {"i:cur", "s:cur", "s:8x16"},
	256:     {"i:cur", "s:rule", "s:4x8x8"},
	512:     {"i:cur", "s:cur", "s:rule"},
	1024:    {"i:cur", "s:cur", "s:rule", "s:8x4x4x8"},
	2048:    {"i:cur", "s:cur", "s:rule", "s:4x8x4x16"},
	4096:    {"i:cur", "s:cur", "s:rule", "s:4x8x4x8x4"},
	8192:    {"i:cur", "s:rule", "s:4x4x4x4x8x4"},
	16384:   {"i:cur", "s:cur", "s:rule", "s:4x8x8x4x4x4"},
	32768:   {"i:cur", "s:rule", "s:8x8x8x4x4x4"},
	65536:   {"i:cur", "s:cur", "s:rule"},
	1 << 17: {"i:cur", "s:rule"},
	1 << 18: {"i:cur", "s:cur", "s:rule"},
	1 << 20: {"i:cur", "s:cur", "s:rule"},
	1000:    {"i:cur", "s:cur"},
	1008:    {"i:cur", "s:cur"},
	1080:    {"i:cur", "s:cur"},
	1296:    {"i:cur", "s:cur"},
	1920:    {"i:cur", "s:cur"},
	2000:    {"i:cur", "s:cur"},
	6000:    {"i:cur", "s:cur"},
	20480:   {"i:cur", "s:cur"},
	45000:   {"i:cur", "s:cur"},
}

// TestSplitAB times splitABSets' factorizations in interleaved rounds, the
// order rotated every round, and prints each one's median time and its
// ratio to the first, with the max/min spread of the per-round ratios.
func TestSplitAB(t *testing.T) {
	if *splitAB == "" {
		t.Skip("set -split.ab=rounds,ms")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*splitAB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	defer splitBench(false)()
	ns := slices.Sorted(maps.Keys(splitABSets))
	for _, n := range ns {
		set := splitABSets[n]
		plans := make([]*skPlan, len(set))
		for v, name := range set {
			f := skFactorize(n) // "rule": with splitTable
			switch s := name[2:]; s {
			case "rule":
			case "cur": // main's: without splitTable
				saved := splitTable
				splitTable = nil
				f = skFactorize(n)
				splitTable = saved
			default:
				f = nil
				for _, x := range strings.Split(s, "x") {
					r, _ := strconv.Atoi(x)
					f = append(f, r)
				}
			}
			if p := product(f); p != n {
				t.Fatalf("%d: %s multiplies to %d", n, name, p)
			}
			kernels.UseStockhamSplit = name[0] == 's'
			plans[v] = newSKPlanFactors(n, f)
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		// Calibrate the iteration count on the first plan.
		iters := 1
		for {
			t0 := time.Now()
			for range iters {
				plans[0].transform(dst, src, false)
			}
			if time.Since(t0) > time.Duration(ms)*time.Millisecond/4 {
				iters *= 4
				break
			}
			iters *= 2
		}
		times := make([][]float64, len(set))
		for round := range rounds {
			for q := range set {
				v := (q + round) % len(set)
				t0 := time.Now()
				for range iters {
					plans[v].transform(dst, src, false)
				}
				times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(iters))
			}
		}
		base := times[0]
		for v, name := range set {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = base[i] / times[v][i]
			}
			fmt.Printf("AB %d %-22s %s %10.1f ns  base/this %.3f  spread %.3f\n", n, name,
				factorName(plans[v].factors()), median(times[v]), median(ratios), slices.Max(ratios)/slices.Min(ratios))
		}
	}
}

func median(x []float64) float64 {
	y := slices.Sorted(slices.Values(x))
	return y[len(y)/2]
}

// factors lists the plan's radices.
func (p *skPlan) factors() []int {
	var f []int
	for _, st := range p.stages {
		f = append(f, st.r)
	}
	return f
}

func product(f []int) int {
	p := 1
	for _, r := range f {
		p *= r
	}
	return p
}
