//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 29: an Intel CPU with AVX2
// and no AVX-512 (Haswell), where the split layout (Round 23) and AMD's
// composite rule (comp2Factors, Round 26) had never been timed. The powers
// of two use Round 28's TestIntelSweep and TestIntelAB (modes a and s); the
// composites use TestHswComp below and Round 26's TestR26Sweep.

import (
	"flag"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

var hswComp = flag.String("hsw.comp", "", "TestHswComp: rounds,ms,min,max (e.g. 5,40,6,16384)")
var hswList = flag.String("hsw.list", "", "TestHswComp: comma-separated lengths instead of min..max")

// TestHswComp times, for every n = 2^e·3^a·5^b (e >= 1, a+b >= 1) in the
// range, the current factorization (skFactorize: Round 24's Intel rule on an
// Intel CPU) and comp2Factors' (Round 26's AMD rule), each interleaved (i)
// and with the split layout (s), in rotated rounds. It prints each variant's
// median and the median of its per-round ratios to cur/i, with their range.
func TestHswComp(t *testing.T) {
	if *hswComp == "" {
		t.Skip("set -hsw.comp=rounds,ms,min,max")
	}
	var rounds, ms, lo, hi int
	if _, err := fmt.Sscanf(*hswComp, "%d,%d,%d,%d", &rounds, &ms, &lo, &hi); err != nil {
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
	defer func(v bool) { kernels.UseStockhamSplit = v }(kernels.UseStockhamSplit)
	for _, n := range ns {
		cur, c2 := skFactorize(n), comp2Factors(n, true)
		if c2 == nil {
			c2 = cur
		}
		type variant struct {
			name string
			p    *skPlan
		}
		var vs []variant
		for _, f := range [][]int{cur, c2} {
			if len(vs) > 0 && slices.Equal(f, cur) {
				continue
			}
			for _, on := range []bool{false, true} {
				kernels.UseStockhamSplit = on
				tag := map[bool]string{false: "i", true: "s"}[on]
				vs = append(vs, variant{factorName(f) + "/" + tag, newSKPlanFactors(n, f)})
			}
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
		fmt.Printf("HC %d%s\n", n, b.String())
	}
}
