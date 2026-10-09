//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 31 (Cascade Lake): where the
// 2-D 1024² transform spends its time, axis by axis, with the row plan as a
// variant.

import (
	"flag"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/go-fft/fft/internal/kernels"
)

var (
	r31Rows    = flag.String("r31.rows", "", "TestR31Rows2D: rounds,iters,n (e.g. 9,8,1024)")
	r31RowVars = flag.String("r31.rowvars", "z:cur,w:4x4x8x8", "TestR31Rows2D: row plan variants (intelVariant names)")
	r31RowSpan = flag.Int("r31.span", 0, "TestR31Rows2D: rows per sweep (0: every row of the m×n array)")
	r31RowM    = flag.Int("r31.m", 0, "TestR31Rows2D: rows of the array (0: n)")
)

// TestR31Rows2D times, for an n×n PlanN, its column step (the strips, src to
// dst) and then its row step (every row in place on dst, as contiguousLines
// runs it) with each row plan variant, in rounds whose variant order rotates.
// The columns run before every row sweep, so the rows meet the cache the
// columns leave, as inside FFT. With -r31.span > 0 the row sweep covers only
// that many rows, repeated to n rows' worth, so they run from cache.
func TestR31Rows2D(t *testing.T) {
	if *r31Rows == "" {
		t.Skip("set -r31.rows=rounds,iters,n")
	}
	var rounds, iters, n int
	if _, err := fmt.Sscanf(*r31Rows, "%d,%d,%d", &rounds, &iters, &n); err != nil {
		t.Fatal(err)
	}
	names := strings.Split(*r31RowVars, ",")
	plans := make([]*skPlan, len(names))
	for v, name := range names {
		var err error
		if plans[v], err = intelVariant(n, name); err != nil {
			t.Fatal(err)
		}
	}
	m := n
	if *r31RowM > 0 {
		m = *r31RowM
	}
	pn := NewPlanN(m, n)
	src := benchComplex(m * n)
	dst := make([]complex128, m*n)
	span := m
	if *r31RowSpan > 0 {
		span = *r31RowSpan
	}
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
				pn.transformAxis(dst, src, 0, false)
				t1 := time.Now()
				for c := range m {
					r := c % span
					p.run(dst[r*n:(r+1)*n], dst[r*n:(r+1)*n], *bp, false)
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
		fmt.Printf("R31 %dx%d span %d %-16s cols %9.0f ns  rows %9.0f ns (%.2f ns/pt)  first/this %.3f  range %.3f-%.3f\n",
			m, n, span, name, median(cols[v]), median(rows[v]), median(rows[v])/float64(m*n),
			median(ratios), slices.Min(ratios), slices.Max(ratios))
	}
}

var r31Comp = flag.String("r31.comp", "", "TestR31Comp: rounds,ms,min,max (e.g. 5,40,6,16384); -hsw.list overrides min..max")

// TestR31Comp is Round 29's TestHswComp for an AVX-512 Intel CPU: for every
// n = 2^e·3^a·5^b (e >= 1, a+b >= 1) in the range, Round 24's Intel rule
// (r24, the reference, what Cascade Lake runs), hswComp2Factors' (hsw) and
// comp2Factors' (c2) factorizations interleaved, and hsw with the 256-bit
// split layout on its radix-4/8 runs (s256: the 512-bit layout, which never
// takes a composite, turned off), in rotated rounds.
func TestR31Comp(t *testing.T) {
	if *r31Comp == "" {
		t.Skip("set -r31.comp=rounds,ms,min,max")
	}
	var rounds, ms, lo, hi int
	if _, err := fmt.Sscanf(*r31Comp, "%d,%d,%d,%d", &rounds, &ms, &lo, &hi); err != nil {
		t.Fatal(err)
	}
	var ns []int
	if *hswList != "" {
		for _, s := range strings.Split(*hswList, ",") {
			var n int
			if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
				t.Fatal(err)
			}
			ns = append(ns, n)
		}
	} else {
		for n := lo; n <= hi; n++ {
			if n%2 == 0 && n&(n-1) != 0 && comp2Factors(n, true) != nil {
				ns = append(ns, n)
			}
		}
	}
	defer func(v, v512 bool) {
		kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = v, v512
	}(kernels.UseStockhamSplit, kernels.UseStockhamSplit512)
	for _, n := range ns {
		r24, c2, hsw := hswR24(n), comp2Factors(n, true), hswComp2Factors(n, true)
		if c2 == nil {
			c2, hsw = r24, r24
		}
		type variant struct {
			name string
			p    *skPlan
		}
		var vs []variant
		add := func(tag string, f []int, split bool) {
			kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = split, !split
			mode := "i"
			if split {
				mode = "s256"
			}
			vs = append(vs, variant{tag + ":" + factorName(f) + "/" + mode, newSKPlanFactors(n, f)})
		}
		add("r24", r24, false)
		add("hsw", hsw, false)
		add("hsw", hsw, true)
		if !slices.Equal(c2, hsw) {
			add("c2", c2, false)
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		fns := make([]func(), len(vs))
		for v := range vs {
			p := vs[v].p
			fns[v] = func() { p.transform(dst, src, false) }
		}
		times := comp2Rotate(fns, rounds, ms)
		var b strings.Builder
		for v := range vs {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			fmt.Fprintf(&b, " %s=%.0f/%.3f/%.3f-%.3f", vs[v].name, median(times[v]), median(ratios),
				slices.Min(ratios), slices.Max(ratios))
		}
		fmt.Printf("CC %d%s\n", n, b.String())
	}
}

var (
	r31Whole  = flag.String("r31.whole", "", "TestR31Whole: rounds,iters")
	r31Shapes = flag.String("r31.shapes", "1024x1024,512x1024,256x1024", "TestR31Whole: shapes")
)

// TestR31Whole times whole N-D transforms with clRowOrder off (main's rows,
// the reference) and on, the two plans sharing one src and one dst, in
// rotated rounds: the change alone, without the allocation placement that
// separates two binaries.
func TestR31Whole(t *testing.T) {
	if *r31Whole == "" {
		t.Skip("set -r31.whole=rounds,iters")
	}
	var rounds, iters int
	if _, err := fmt.Sscanf(*r31Whole, "%d,%d", &rounds, &iters); err != nil {
		t.Fatal(err)
	}
	for _, s := range strings.Split(*r31Shapes, ",") {
		var shape []int
		for _, x := range strings.Split(s, "x") {
			var n int
			if _, err := fmt.Sscanf(x, "%d", &n); err != nil {
				t.Fatal(err)
			}
			shape = append(shape, n)
		}
		saved := clRowOrder
		clRowOrder = nil
		pm := NewPlanN(shape...)
		clRowOrder = saved
		pb := NewPlanN(shape...)
		size := pm.Len()
		src := benchComplex(size)
		dst := make([]complex128, size)
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
		fmt.Printf("WHOLE %-12s main %10.0f ns  rows %v %10.0f ns  main/this %.3f  range %.3f-%.3f\n", s,
			median(times[0]), factorsOf(pb.axes[len(shape)-1]), median(times[1]), median(ratios), slices.Min(ratios), slices.Max(ratios))
	}
}

func factorsOf(p *Plan) []int {
	var f []int
	if p.sk != nil {
		for _, st := range p.sk.stages {
			f = append(f, st.r)
		}
	}
	return f
}
