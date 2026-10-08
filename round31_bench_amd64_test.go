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
