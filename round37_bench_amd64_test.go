//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 37 on Haswell: the
// contiguous last axis of large N-D plans timed alone with each row plan as
// a variant (Round 31's TestR31Rows2D for any number of axes), and the whole
// transform with a row order against the 1-D plan's.

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
	hwRows     = flag.String("hw.rows", "", "TestHwRows: rounds,iters (e.g. 9,4)")
	hwShapes   = flag.String("hw.shapes", "1024x1024,512x1024,4x256x1024", "TestHwRows / TestHwWhole: shapes")
	hwRowVars  = flag.String("hw.rowvars", "s:cur,s:4x8x4x8", "TestHwRows: row plan variants (intelVariant names; the first is the reference)")
	hwWhole    = flag.String("hw.whole", "", "TestHwWhole: rounds,iters")
	hwRowOrder = flag.String("hw.roworder", "4x8x4x8", "TestHwWhole: the factorization the last axis takes")
)

func hwShape(t *testing.T, s string) []int {
	var shape []int
	for _, x := range strings.Split(s, "x") {
		n, err := strconv.Atoi(x)
		if err != nil {
			t.Fatal(err)
		}
		shape = append(shape, n)
	}
	return shape
}

// TestHwRows times, for each shape, every axis but the last (src to dst,
// then in place, as PlanN.transform runs them) and then the last axis, every
// row in place on dst, with each row plan variant, in rounds whose variant
// order rotates; the rows meet the cache the other axes leave.
func TestHwRows(t *testing.T) {
	if *hwRows == "" {
		t.Skip("set -hw.rows=rounds,iters")
	}
	var rounds, iters int
	if _, err := fmt.Sscanf(*hwRows, "%d,%d", &rounds, &iters); err != nil {
		t.Fatal(err)
	}
	for _, s := range strings.Split(*hwShapes, ",") {
		shape := hwShape(t, s)
		n := shape[len(shape)-1]
		names := strings.Split(*hwRowVars, ",")
		plans := make([]*skPlan, len(names))
		for v, name := range names {
			var err error
			if plans[v], err = intelVariant(n, name); err != nil {
				t.Fatal(err)
			}
		}
		pn := NewPlanN(shape...)
		src := benchComplex(pn.size)
		dst := make([]complex128, pn.size)
		lines := pn.size / n
		cols := make([][]float64, len(plans))
		rows := make([][]float64, len(plans))
		for round := range rounds {
			for q := range plans {
				v := (q + round) % len(plans)
				p := plans[v]
				bp := p.scratch.Get().(*[]complex128)
				var tc, tr time.Duration
				for range iters {
					t0 := time.Now()
					in := src
					for ax := range len(shape) - 1 {
						pn.transformAxis(dst, in, ax, false)
						in = dst
					}
					t1 := time.Now()
					for c := range lines {
						p.run(dst[c*n:(c+1)*n], dst[c*n:(c+1)*n], *bp, false)
					}
					tc += t1.Sub(t0)
					tr += time.Since(t1)
				}
				p.scratch.Put(bp)
				cols[v] = append(cols[v], float64(tc.Nanoseconds())/float64(iters))
				rows[v] = append(rows[v], float64(tr.Nanoseconds())/float64(iters))
			}
		}
		for v, name := range names {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = rows[0][i] / rows[v][i]
			}
			var f []int
			for _, st := range plans[v].stages {
				f = append(f, st.r)
			}
			fmt.Printf("HWROWS %-12s %-14s %-12s other axes %10.0f ns  rows %10.0f ns (%.2f ns/pt)  first/this %.3f  range %.3f-%.3f\n",
				s, name, factorName(f), median(cols[v]), median(rows[v]), median(rows[v])/float64(pn.size),
				median(ratios), slices.Min(ratios), slices.Max(ratios))
		}
	}
}

// TestHwWhole times whole N-D transforms with the last axis on its 1-D plan
// (the reference) and on -hw.roworder's factorization through clRowOrder,
// the two plans sharing one src and one dst, in rotated rounds.
func TestHwWhole(t *testing.T) {
	if *hwWhole == "" {
		t.Skip("set -hw.whole=rounds,iters")
	}
	var rounds, iters int
	if _, err := fmt.Sscanf(*hwWhole, "%d,%d", &rounds, &iters); err != nil {
		t.Fatal(err)
	}
	f := hwShape(t, *hwRowOrder)
	for _, s := range strings.Split(*hwShapes, ",") {
		shape := hwShape(t, s)
		n := shape[len(shape)-1]
		savedOrder, savedCache := clRowOrder, clRowCache
		clRowOrder, clRowCache = nil, map[int]*Plan{}
		pm := NewPlanN(shape...)
		clRowOrder = map[int][]int{n: f}
		pb := NewPlanN(shape...)
		clRowOrder, clRowCache = savedOrder, savedCache
		src := benchComplex(pm.Len())
		dst := make([]complex128, pm.Len())
		fns := []func(){func() { pm.FFT(dst, src) }, func() { pb.FFT(dst, src) }}
		times := make([][]float64, 2)
		for round := range rounds {
			for q := range fns {
				v := (q + round) % 2
				t0 := time.Now()
				for range iters {
					fns[v]()
				}
				times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(iters))
			}
		}
		ratios := make([]float64, rounds)
		for i := range ratios {
			ratios[i] = times[0][i] / times[1][i]
		}
		fmt.Printf("HWWHOLE %-12s main %v %10.0f ns  rows %v %10.0f ns  main/this %.3f  range %.3f-%.3f\n", s,
			factorsOf(pm.axes[len(shape)-1]), median(times[0]), factorsOf(pb.axes[len(shape)-1]), median(times[1]),
			median(ratios), slices.Min(ratios), slices.Max(ratios))
	}
}

var (
	hwE11     = flag.String("hw.e11", "", "TestHwE11: rounds,ms")
	hwE11List = flag.String("hw.e11list", "6144,10240,12288", "TestHwE11: lengths")
)

// hwE11Candidates lists, for n = 2^e·3^a·5^b, the current factorization
// first, then comp2Candidates' shapes (the 3s and 5s, a power-of-two tail,
// the 10, 12 and 20 passes) with every ordering of radix-4, -8 and -16
// passes as the tail.
func hwE11Candidates(n int) [][]int {
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
					for _, tail := range intelCompositions(ep) {
						f := rep(rep(rep(nil, 3, a-t12-t15), 5, b-t20-t10-t15), 15, t15)
						f = append(f, tail...)
						f = rep(rep(rep(f, 10, t10), 12, t12), 20, t20)
						if k := factorName(f); product(f) == n && !seen[k] {
							seen[k] = true
							out = append(out, f)
						}
					}
				}
			}
		}
	}
	return out
}

// TestHwE11 times hwE11Candidates for each -hw.e11list length in rotated
// rounds and prints them fastest first, each as current time ÷ its time.
func TestHwE11(t *testing.T) {
	if *hwE11 == "" {
		t.Skip("set -hw.e11=rounds,ms")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*hwE11, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	for _, s := range strings.Split(*hwE11List, ",") {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		cs := hwE11Candidates(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		for _, f := range cs {
			p := newSKPlanFactors(n, f)
			fns = append(fns, func() { p.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		type row struct {
			f      string
			ns, rr float64
			lo, hi float64
		}
		var rows []row
		for v, f := range cs {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			rows = append(rows, row{factorName(f), median(times[v]), median(ratios), slices.Min(ratios), slices.Max(ratios)})
		}
		slices.SortFunc(rows, func(x, y row) int {
			if x.rr > y.rr {
				return -1
			}
			if x.rr < y.rr {
				return 1
			}
			return 0
		})
		fmt.Printf("E11 %d current %s, %d candidates\n", n, factorName(cs[0]), len(cs))
		for i, r := range rows {
			if i < 12 || r.f == factorName(cs[0]) {
				fmt.Printf("E11 %d %-20s %10.1f ns  current/this %.3f (%.3f-%.3f)\n", n, r.f, r.ns, r.rr, r.lo, r.hi)
			}
		}
	}
}
