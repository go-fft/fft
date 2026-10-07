//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 29: an Intel CPU with AVX2
// and no AVX-512 (Haswell), where the split layout (Round 23) and AMD's
// composite rule (comp2Factors, Round 26) had never been timed. The powers
// of two use Round 28's TestIntelSweep and TestIntelAB (modes a and s); the
// composites use TestHswComp below and Round 26's TestR26Sweep.

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

var hswComp = flag.String("hsw.comp", "", "TestHswComp: rounds,ms,min,max (e.g. 5,40,6,16384)")
var hswList = flag.String("hsw.list", "", "TestHswComp: comma-separated lengths instead of min..max")

// hswR24 is Round 24's Intel composite rule for n (radix-12 passes, else a
// radix-16 tail, else skFactorizeOrder's), what Haswell ran before Round 29.
func hswR24(n int) []int {
	if f := comp2IntelRule(n); f != nil {
		return f
	}
	return skFactorizeOrder(n, true)
}

// TestHswComp times, for every n = 2^e·3^a·5^b (e >= 1, a+b >= 1) in the
// range, Round 24's Intel rule (r24, interleaved: the reference),
// comp2Factors' (c2, Round 26's AMD rule) and hswComp2Factors' (hsw)
// factorizations, interleaved (i), with the split layout (s) and with the
// split layout for runs of two passes or more only (s1), in rotated rounds.
// It prints each variant's median and the median of its per-round ratios to
// r24/i, with their range.
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
	defer func(v, w bool) { kernels.UseStockhamSplit, kernels.UseStockhamSplitSingle = v, w }(kernels.UseStockhamSplit, kernels.UseStockhamSplitSingle)
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
		add := func(tag string, f []int, mode string) {
			kernels.UseStockhamSplit = mode != "i"
			kernels.UseStockhamSplitSingle = mode != "s1"
			vs = append(vs, variant{tag + ":" + factorName(f) + "/" + mode, newSKPlanFactors(n, f)})
		}
		add("r24", r24, "i")
		add("hsw", hsw, "i")
		add("hsw", hsw, "s")
		add("hsw", hsw, "s1")
		if !slices.Equal(c2, hsw) {
			add("c2", c2, "i")
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

var hswExport = flag.String("hsw.export", "", "TestHswExport: directory the inputs and outputs go to")

// TestHswExport writes, for each -hsw.list length, a random input and the
// forward transforms of Round 24's Intel rule (cur) and hswComp2Factors'
// (c2) as little-endian complex128, so numpy can judge both on the bytes
// this code received.
func TestHswExport(t *testing.T) {
	if *hswExport == "" || *hswList == "" {
		t.Skip("set -hsw.export=dir and -hsw.list")
	}
	r := rand.New(rand.NewPCG(29, 2026))
	write := func(name string, x []complex128) {
		b := make([]byte, 0, 16*len(x))
		for _, v := range x {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(real(v)))
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(imag(v)))
		}
		if err := os.WriteFile(filepath.Join(*hswExport, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range strings.Split(*hswList, ",") {
		var n int
		if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
			t.Fatal(err)
		}
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		write(fmt.Sprintf("in_%d.bin", n), src)
		for tag, f := range map[string][]int{"cur": hswR24(n), "c2": hswComp2Factors(n, true)} {
			if f == nil {
				continue
			}
			dst := make([]complex128, n)
			newSKPlanFactors(n, f).transform(dst, src, false)
			write(fmt.Sprintf("%s_%d.bin", tag, n), dst)
			fmt.Printf("EXPORT %d %s %s\n", n, tag, factorName(f))
		}
	}
}
