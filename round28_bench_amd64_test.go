//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 28: on an AVX-512 Intel CPU,
// the AVX-512 interleaved passes against the AVX2 passes, interleaved and
// split (Round 23's layout), and the radix-16 pass, from 64 to 16384 points.

import (
	"flag"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-fft/fft/internal/kernels"
)

var (
	intelSweep = flag.String("intel.sweep", "", "TestIntelSweep: rounds,ms,minE,maxE (e.g. 3,20,6,14)")
	intelAB    = flag.String("intel.ab", "", "TestIntelAB: rounds,ms (e.g. 7,40) over -intel.set")
	intelSet   = flag.String("intel.set", "", "TestIntelAB: n=variant,variant;n=... (variant mode:factors)")
	intelModes = flag.String("intel.modes", "all", "TestIntelSweep: all, or w and z only")
)

// intelVariant builds the plan a variant names. mode is one of
//
//	w   interleaved, the AVX-512 kernels where wide512 allows them (main's)
//	a   interleaved, AVX2 only
//	s   split runs (Round 23), every interleaved pass AVX2
//	sw  split runs, the interleaved passes (the final one) AVX-512 where wide512 allows
//	z   split runs at 512 bits (Round 28), the interleaved passes as w
//
// and factors is "cur" (skFactorize on this machine), "r23" (the split table
// Round 23 chose on Zen 3, else skFactorize) or radices joined by 'x'.
func intelVariant(n int, name string) (*skPlan, error) {
	mode, fs, ok := strings.Cut(name, ":")
	if !ok {
		return nil, fmt.Errorf("variant %q: want mode:factors", name)
	}
	var f []int
	switch fs {
	case "cur":
		f = skFactorize(n)
	case "r23":
		f = splitTableAMD64(true)[n]
		if f == nil {
			f = skFactorize(n)
		}
	default:
		for _, x := range strings.Split(fs, "x") {
			r, err := strconv.Atoi(x)
			if err != nil {
				return nil, err
			}
			f = append(f, r)
		}
	}
	if product(f) != n {
		return nil, fmt.Errorf("%d: %s multiplies to %d", n, name, product(f))
	}
	old, old512, oldFloor := kernels.UseStockhamSplit, kernels.UseStockhamSplit512, hswSplitFloor
	defer func() { kernels.UseStockhamSplit, kernels.UseStockhamSplit512, hswSplitFloor = old, old512, oldFloor }()
	kernels.UseStockhamSplit = mode == "s" || mode == "sw"
	hswSplitFloor = 0 // every length its mode says, whatever Round 29's floor
	kernels.UseStockhamSplit512 = mode == "z"
	p := newSKPlanFactors(n, f)
	for k := range p.stages {
		switch mode {
		case "w", "sw", "z":
		case "a", "s":
			p.stages[k].wide = false
		default:
			return nil, fmt.Errorf("variant %q: unknown mode", name)
		}
	}
	return p, nil
}

// intelCompositions lists every ordered factorization of 2^e into radices 4,
// 8 and 16.
func intelCompositions(e int) [][]int {
	if e == 0 {
		return [][]int{nil}
	}
	var out [][]int
	for _, b := range []int{2, 3, 4} {
		if b > e {
			continue
		}
		for _, rest := range intelCompositions(e - b) {
			out = append(out, append([]int{1 << b}, rest...))
		}
	}
	return out
}

// intelTime times the plans in rounds, the order rotated every round, and
// returns each plan's per-round ns per transform.
func intelTime(plans []*skPlan, n, rounds, ms int) [][]float64 {
	src := benchComplex(n)
	dst := make([]complex128, n)
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
	times := make([][]float64, len(plans))
	for round := range rounds {
		for q := range plans {
			v := (q + round) % len(plans)
			t0 := time.Now()
			for range iters {
				plans[v].transform(dst, src, false)
			}
			times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(iters))
		}
	}
	return times
}

// TestIntelSweep times, for every power of two 2^minE .. 2^maxE, every
// factorization into radices 4, 8 and 16 in modes w, s and sw (a for the
// ones with radix 16, which has no 512-bit kernel), and prints them sorted by
// median time with the ratio to w:cur.
func TestIntelSweep(t *testing.T) {
	if *intelSweep == "" {
		t.Skip("set -intel.sweep=rounds,ms,minE,maxE")
	}
	var rounds, ms, minE, maxE int
	if _, err := fmt.Sscanf(*intelSweep, "%d,%d,%d,%d", &rounds, &ms, &minE, &maxE); err != nil {
		t.Fatal(err)
	}
	for e := minE; e <= maxE; e++ {
		n := 1 << e
		names := []string{"w:cur", "a:cur"}
		for _, f := range intelCompositions(e) {
			fs := factorName(f)
			names = append(names, "w:"+fs)
			if *intelModes == "all" {
				names = append(names, "s:"+fs)
				if wide512(n) {
					names = append(names, "sw:"+fs, "a:"+fs)
				}
			}
			if wide512(n) {
				names = append(names, "z:"+fs)
			}
		}
		intelReport(t, n, names, rounds, ms)
	}
}

func intelReport(t *testing.T, n int, names []string, rounds, ms int) {
	plans := make([]*skPlan, len(names))
	for v, name := range names {
		p, err := intelVariant(n, name)
		if err != nil {
			t.Fatal(err)
		}
		plans[v] = p
	}
	times := intelTime(plans, n, rounds, ms)
	type row struct {
		name           string
		med, r, spread float64
	}
	var rows []row
	for v, name := range names {
		ratios := make([]float64, rounds)
		for i := range ratios {
			ratios[i] = times[0][i] / times[v][i]
		}
		rows = append(rows, row{name, median(times[v]), median(ratios), slices.Max(ratios) / slices.Min(ratios)})
	}
	slices.SortStableFunc(rows, func(a, b row) int {
		switch {
		case a.med < b.med:
			return -1
		case a.med > b.med:
			return 1
		}
		return 0
	})
	for _, r := range rows {
		fmt.Printf("SW %d %-20s %10.1f ns  first/this %.3f  spread %.3f\n", n, r.name, r.med, r.r, r.spread)
	}
}

// TestIntelAB times -intel.set's variants per length in rotated rounds and
// prints each one's median and its ratio to the first (median of per-round
// ratios), with their spread.
func TestIntelAB(t *testing.T) {
	if *intelAB == "" || *intelSet == "" {
		t.Skip("set -intel.ab=rounds,ms and -intel.set")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*intelAB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	for _, grp := range strings.Split(*intelSet, ";") {
		ns, vs, _ := strings.Cut(grp, "=")
		n, err := strconv.Atoi(ns)
		if err != nil {
			t.Fatal(err)
		}
		names := strings.Split(vs, ",")
		plans := make([]*skPlan, len(names))
		for v, name := range names {
			if plans[v], err = intelVariant(n, name); err != nil {
				t.Fatal(err)
			}
		}
		times := intelTime(plans, n, rounds, ms)
		for v, name := range names {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			fmt.Printf("AB %d %-20s %10.1f ns  first/this %.3f  range %.3f-%.3f\n", n, name,
				median(times[v]), median(ratios), slices.Min(ratios), slices.Max(ratios))
		}
	}
}
