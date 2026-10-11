package fft

import (
	"encoding/binary"
	"flag"
	"fmt"
	"github.com/go-fft/fft/internal/kernels"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

// Round 35 (Cascade Lake): the rows where FFTW still leads, located part by
// part. Every helper here is prefixed cs.

var (
	csAB = flag.String("cs.ab", "", "rounds,ms: run Round 35's in-process tests")
)

func csRounds(t *testing.T) (rounds, ms int) {
	if *csAB == "" {
		t.Skip("set -cs.ab=rounds,ms")
	}
	if _, err := fmt.Sscanf(*csAB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	return rounds, ms
}

func csInts(t *testing.T, s string) []int {
	var out []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(f)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, n)
	}
	return out
}

// csCascadeParts is cascadeRun out of place with each part switchable: the
// passes over the whole array (pair), the group passes before the last, the
// final runs.
func (p *skPlan) csCascadeParts(dst, src, buf []complex128, pair, groups, last bool) {
	n, t, b := p.n, p.cascT, p.cascB
	scr := offTheSets(buf[:n+setSpan], dst, n)
	loc := buf[n+setSpan:]
	in, k := src, 0
	if p.cascPair {
		out := scr
		if t%2 == 1 {
			out = dst
		}
		if pair {
			p.cascadePair(out, in, loc, false)
		}
		in, k = out, 2
	}
	for ; k < t; k++ {
		out := scr
		if (t-1-k)%2 == 1 {
			out = dst
		}
		if pair {
			p.stages[k].pass(out, in, false)
		}
		in = out
	}
	s := len(p.stages)
	l1t := p.stages[t].l1
	m := n / l1t
	g := b * m
	bufs := [2][]complex128{loc[:g], loc[g+cascadeSkew : 2*g+cascadeSkew]}
	lst := &p.stages[s-1]
	for k0 := 0; k0 < l1t; k0 += b {
		cur := in[k0*m : k0*m+g]
		l1 := b
		for q := t; q < s-1; q++ {
			st := &p.stages[q]
			out := bufs[(q-t)%2]
			if groups {
				st.cascadePass(out, cur, l1, false)
			}
			l1 *= st.r
			cur = out
		}
		if last {
			lst.cascadeLast(dst[k0:], cur, n/lst.r, b, l1t, false)
		}
	}
}

// TestCsCascadeParts times, for each -r26.sets factorization, the plan's transform,
// the same factorization breadth first (with the layout the plan would have
// without the blocked schedule), and the blocked schedule's three parts
// alone, in rotated rounds.
func TestCsCascadeParts(t *testing.T) {
	rounds, ms := csRounds(t)
	ns, sets := comp2Variants(t)
	for _, n := range ns {
		for _, f := range sets[n] {
			p := newSKPlanFactors(n, f)
			saved := cascadeMin
			cascadeMin = 0
			bf := newSKPlanFactors(n, f)
			cascadeMin = saved
			src := benchComplex(n)
			dst := make([]complex128, n)
			buf := make([]complex128, cascadeScratch(n))
			fns := []func(){
				func() { p.transform(dst, src, false) },
				func() { bf.transform(dst, src, false) },
			}
			if p.cascT > 0 {
				fns = append(fns,
					func() { p.csCascadeParts(dst, src, buf, true, false, false) },
					func() { p.csCascadeParts(dst, src, buf, false, true, false) },
					func() { p.csCascadeParts(dst, src, buf, false, false, true) },
				)
			}
			times := comp2Rotate(fns, rounds, ms)
			var modes []string
			for _, st := range bf.stages {
				modes = append(modes, strconv.Itoa(int(st.split)))
			}
			fmt.Printf("CASC %d %s t=%d b=%d pair=%v whole %.1f ns/pt | breadth(split %s) %.1f",
				n, factorName(f), p.cascT, p.cascB, p.cascPair, median(times[0])/float64(n), strings.Join(modes, ""), median(times[1])/float64(n))
			if p.cascT > 0 {
				fmt.Printf(" | pair %.2f groups %.2f last %.2f", median(times[2])/float64(n), median(times[3])/float64(n), median(times[4])/float64(n))
			}
			fmt.Println()
		}
	}
}

// TestCsOrders times -r26.sets' factorizations of each length, each built as
// a plan of its own (blocked schedule and layout as the plan builder gives
// them), against the first, in rotated rounds; per variant its shape and
// median ns per point.
func TestCsOrders(t *testing.T) {
	rounds, ms := csRounds(t)
	ns, sets := comp2Variants(t)
	for _, n := range ns {
		var fns []func()
		var ps []*skPlan
		src := benchComplex(n)
		dst := make([]complex128, n)
		for _, f := range sets[n] {
			for _, cm := range csCascadeMins() {
				saved := cascadeMin
				if cm >= 0 {
					cascadeMin = cm
				}
				p := newSKPlanFactors(n, f)
				cascadeMin = saved
				ps = append(ps, p)
				fns = append(fns, func() { p.transform(dst, src, false) })
			}
		}
		times := comp2Rotate(fns, rounds, ms)
		base := median(times[0])
		for k, p := range ps {
			var modes []string
			for _, st := range p.stages {
				modes = append(modes, strconv.Itoa(int(st.split)))
			}
			lo, hi := 1e300, 0.0
			for r := range times[k] {
				q := times[0][r] / times[k][r]
				lo, hi = min(lo, q), max(hi, q)
			}
			fmt.Printf("ORD %d %-22s t=%d b=%d pair=%v split=%s %.2f ns/pt  first/this %.3f [%.3f %.3f]\n",
				n, factorName(p.factors()), p.cascT, p.cascB, p.cascPair, strings.Join(modes, ""), median(times[k])/float64(n), base/median(times[k]), lo, hi)
		}
	}
}

// csPow2Sets is every ordering of radix-4 and radix-8 passes whose product
// is 2^e.
func csPow2Sets(e int) [][]int {
	var out [][]int
	seen := map[string]bool{}
	for a := 0; 3*a <= e; a++ {
		if (e-3*a)%2 != 0 {
			continue
		}
		var f []int
		for range a {
			f = append(f, 8)
		}
		for range (e - 3*a) / 2 {
			f = append(f, 4)
		}
		for _, q := range perms(f) {
			if k := factorName(q); !seen[k] {
				seen[k] = true
				out = append(out, q)
			}
		}
	}
	return out
}

var csSweepE = flag.String("cs.sweep", "16", "TestCsSweep: exponents")

// TestCsSweep times the rule against every ordering of radix-4 and -8 passes
// for each 2^e of -cs.sweep (at most -cs.maxpasses passes), in rotated rounds.
func TestCsSweep(t *testing.T) {
	rounds, ms := csRounds(t)
	for _, e := range csInts(t, *csSweepE) {
		n := 1 << e
		sets := append([][]int{skFactorize(n)}, csPow2Sets(e)...)
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		var ps []*skPlan
		for _, f := range sets {
			if len(f) > *csMaxPasses {
				continue
			}
			p := newSKPlanFactors(n, f)
			ps = append(ps, p)
			fns = append(fns, func() { p.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		base := median(times[0])
		for k, p := range ps {
			fmt.Printf("SWEEP %d %-22s t=%d b=%d pair=%v %.2f ns/pt rule/this %.3f\n",
				n, factorName(p.factors()), p.cascT, p.cascB, p.cascPair, median(times[k])/float64(n), base/median(times[k]))
		}
	}
}

var csMaxPasses = flag.Int("cs.maxpasses", 7, "TestCsSweep: longest factorization")

// csGroupsSplit is the groups-and-last part of csCascadeParts with the
// group passes before the last in the given layout modes and twiddles.
func (p *skPlan) csGroupsSplit(dst, in, buf []complex128, modes []uint8, tws [][]complex128) {
	n, t, b := p.n, p.cascT, p.cascB
	loc := buf[n+setSpan:]
	s := len(p.stages)
	l1t := p.stages[t].l1
	m := n / l1t
	g := b * m
	bufs := [2][]complex128{loc[:g], loc[g+cascadeSkew : 2*g+cascadeSkew]}
	lst := &p.stages[s-1]
	for k0 := 0; k0 < l1t; k0 += b {
		cur := in[k0*m : k0*m+g]
		l1 := b
		for q := t; q < s-1; q++ {
			st := &p.stages[q]
			out := bufs[(q-t)%2]
			if !kernels.StockhamPassLayout(modes[q-t], st.r, st.ido, l1, cur, out, tws[q-t], false, true) {
				panic("no kernel")
			}
			l1 *= st.r
			cur = out
		}
		lst.cascadeLast(dst[k0:], cur, n/lst.r, b, l1t, false)
	}
}

// TestCsSplitGroups times the groups of the blocked schedule interleaved
// (as main runs them) and in the 512-bit split layout, after checking the
// two give the same bits.
func TestCsSplitGroups(t *testing.T) {
	rounds, ms := csRounds(t)
	ns, sets := comp2Variants(t)
	for _, n := range ns {
		for _, f := range sets[n] {
			p := newSKPlanFactors(n, f)
			if p.cascT == 0 {
				continue
			}
			root := twiddleTable(n)
			var rs, idos, l1s []int
			for _, st := range p.stages[p.cascT : len(p.stages)-1] {
				rs, idos, l1s = append(rs, st.r), append(idos, st.ido), append(l1s, st.l1)
			}
			modes := kernels.StockhamSplitModes(rs, idos, l1s)
			tws := make([][]complex128, len(modes))
			inter := make([]uint8, len(modes))
			itws := make([][]complex128, len(modes))
			for k, md := range modes {
				st := p.stages[p.cascT+k]
				itws[k] = st.twX
				if md == 0 {
					tws[k] = st.twX
					continue
				}
				tws[k], _ = kernels.StockhamSplitTwiddlesFor(md, st.r, st.ido, st.l1, root)
			}
			src := benchComplex(n)
			buf := make([]complex128, cascadeScratch(n))
			// The input of the groups: what the passes over the array leave.
			in := make([]complex128, n)
			{
				d := make([]complex128, n)
				p.csCascadeParts(d, src, buf, true, false, false)
				scr := offTheSets(buf[:n+setSpan], d, n)
				copy(in, scr) // pass t-1 writes the scratch buffer
			}
			want := make([]complex128, n)
			p.transform(want, src, false)
			a, b := make([]complex128, n), make([]complex128, n)
			p.csGroupsSplit(a, in, buf, inter, itws)
			p.csGroupsSplit(b, in, buf, modes, tws)
			for i := range want {
				if a[i] != want[i] || b[i] != want[i] {
					t.Fatalf("%d %v: [%d] %v %v want %v", n, f, i, a[i], b[i], want[i])
				}
			}
			times := comp2Rotate([]func(){
				func() { p.csGroupsSplit(a, in, buf, inter, itws) },
				func() { p.csGroupsSplit(b, in, buf, modes, tws) },
				func() { p.transform(want, src, false) },
			}, rounds, ms)
			fmt.Printf("GROUPS %d %s modes %v: interleaved %.3f split %.3f ns/pt (whole %.2f), inter/split %.3f\n", n, factorName(f), modes,
				median(times[0])/float64(n), median(times[1])/float64(n), median(times[2])/float64(n), median(times[0])/median(times[1]))
		}
	}
}

var csCmins = flag.String("cs.cmins", "-1", "TestCsOrders: cascadeMin values each variant is built with (-1: the default)")

func csCascadeMins() []int {
	var out []int
	for _, f := range strings.Split(*csCmins, ",") {
		v, _ := strconv.Atoi(f)
		out = append(out, v)
	}
	return out
}

// csBuildWith builds fn's plans with splitTable replaced by tab and the plan
// cache emptied of the lengths built, so each variant gets plans of its own.
func csBuildWith[T any](tab map[int][]int, forget []int, fn func() T) T {
	saved := splitTable
	splitTable = tab
	defer func() { splitTable = saved }()
	for _, n := range forget {
		znForget(n)
	}
	v := fn()
	for _, n := range forget {
		znForget(n)
	}
	return v
}

var csE2E = flag.String("cs.e2e", "C65536,C524288,C1048576,R1048576,I1048576,R131072,C131072,C262144,C32768,C4096,R65536,C1024",
	"TestCsCascadeE2E: rows (C complex, R RFFT, I IRFFT), touched first")

// TestCsCascadeE2E: whole transforms through the exported API, main's table
// (csCascadeOrders removed) against this branch's, both plans in one
// process, rotated every round. Rows the table does not reach build the same
// plan twice: their ratio is the noise.
func TestCsCascadeE2E(t *testing.T) {
	rounds, ms := csRounds(t)
	old := map[int][]int{}
	for n, f := range splitTable {
		if csCascadeOrders()[n] == nil {
			old[n] = f
		}
	}
	for _, row := range strings.Split(*csE2E, ",") {
		n, err := strconv.Atoi(row[1:])
		if err != nil {
			t.Fatal(err)
		}
		var fns []func()
		var names []string
		for _, tab := range []map[int][]int{old, splitTable} {
			switch row[0] {
			case 'C':
				p := csBuildWith(tab, []int{n}, func() *Plan { return NewPlan(n) })
				src, dst := benchComplex(n), make([]complex128, n)
				fns = append(fns, func() { p.FFT(dst, src) })
				names = append(names, factorName(p.sk.factors()))
			case 'R', 'I':
				p := csBuildWith(tab, []int{n / 2}, func() *RealPlan { return NewRealPlan(n) })
				src, dst, back := benchReal(n), make([]complex128, n/2+1), make([]float64, n)
				p.RFFT(dst, src)
				if row[0] == 'R' {
					fns = append(fns, func() { p.RFFT(dst, src) })
				} else {
					fns = append(fns, func() { p.IRFFT(back, dst) })
				}
				names = append(names, factorName(p.half.sk.factors()))
			}
		}
		times := comp2Rotate(fns, rounds, ms)
		lo, hi := 1e300, 0.0
		var qs []float64
		for r := range times[0] {
			q := times[0][r] / times[1][r]
			qs = append(qs, q)
			lo, hi = min(lo, q), max(hi, q)
		}
		fmt.Printf("E2E %-9s main %-16s %10.0f ns  branch %-16s %10.0f ns  main/branch %.3f [%.3f %.3f] per-round median %.3f\n",
			row, names[0], median(times[0]), names[1], median(times[1]), median(times[0])/median(times[1]), lo, hi, median(qs))
	}
}

// csPairW is cascadePair with the chunk width w.
func (p *skPlan) csPairW(out, in, loc []complex128, w int) {
	s0, s1 := &p.stages[0], &p.stages[1]
	r0, r1, ido0, ido1 := s0.r, s1.r, s0.ido, s1.ido
	lb := loc[:r0*r1*w]
	for i0 := 0; i0 < ido1; i0 += w {
		s0.cascadeStrided(lb, in[i0:], w, r1, ido0, ido1, r1*w, w, i0, ido1, false)
		s1.cascadeStrided(out[i0:], lb, w, r0, w, r1*w, r0*ido1, ido1, i0, 0, false)
	}
}

var csChunks = flag.String("cs.chunks", "32,64,128,256,512", "TestCsPairChunk: chunk widths")

// TestCsPairChunk times the fused pair alone at each chunk width, checking
// each gives the bits of the plan's own pair.
func TestCsPairChunk(t *testing.T) {
	rounds, ms := csRounds(t)
	ns, sets := comp2Variants(t)
	for _, n := range ns {
		for _, f := range sets[n] {
			p := newSKPlanFactors(n, f)
			if !p.cascPair {
				continue
			}
			src := benchComplex(n)
			want, got := make([]complex128, n), make([]complex128, n)
			loc := make([]complex128, 4*cascadeLocal)
			p.cascadePair(want, src, loc, false)
			var fns []func()
			ws := csInts(t, *csChunks)
			for _, w := range ws {
				p.csPairW(got, src, loc, w)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("%d w=%d: [%d] differs", n, w, i)
					}
				}
				fns = append(fns, func() { p.csPairW(got, src, loc, w) })
			}
			times := comp2Rotate(fns, rounds, ms)
			var b strings.Builder
			for k, w := range ws {
				fmt.Fprintf(&b, " w%d %.3f", w, median(times[k])/float64(n))
			}
			fmt.Printf("PAIR %d %s ns/pt:%s\n", n, factorName(f), b.String())
		}
	}
}

var csCompMax = flag.Int("cs.compmax", 1<<14, "TestCsComp9: largest length")
var csCompList = flag.String("cs.complist", "", "TestCsComp9: these lengths instead")

// TestCsComp9 times, for every n = 2^e·3^a·5^b up to -cs.compmax (e >= 1)
// whose Intel rule opens with two or more radix-3 passes, the rule without
// radix 9 against csRadix9For's, in rotated rounds, and summarizes.
func TestCsComp9(t *testing.T) {
	rounds, ms := csRounds(t)
	var qs []float64
	var ns []int
	for n := 2; n <= *csCompMax; n++ {
		m := n
		for _, p := range []int{2, 3, 5} {
			for m%p == 0 {
				m /= p
			}
		}
		if m == 1 && n%2 == 0 {
			ns = append(ns, n)
		}
	}
	if *csCompList != "" {
		ns = csInts(t, *csCompList)
	}
	for _, n := range ns {
		f0 := csCompFactorize(n)
		f1 := csRadix9For(f0, true)
		if slices.Equal(f0, f1) {
			continue
		}
		p0, p1 := newSKPlanFactors(n, f0), newSKPlanFactors(n, f1)
		src, dst := benchComplex(n), make([]complex128, n)
		times := comp2Rotate([]func(){
			func() { p0.transform(dst, src, false) },
			func() { p1.transform(dst, src, false) },
		}, rounds, ms)
		lo, hi := 1e300, 0.0
		for r := range times[0] {
			q := times[0][r] / times[1][r]
			lo, hi = min(lo, q), max(hi, q)
		}
		q := median(times[0]) / median(times[1])
		qs = append(qs, q)
		fmt.Printf("COMP9 %6d %-18s %-18s %.3f [%.3f %.3f]\n", n, factorName(f0), factorName(f1), q, lo, hi)
	}
	g, mn, mx := 0.0, 1e300, 0.0
	for _, q := range qs {
		g += math.Log(q)
		mn, mx = min(mn, q), max(mx, q)
	}
	fmt.Printf("COMP9 summary: %d lengths, geometric mean %.3f, min %.3f, max %.3f\n", len(qs), math.Exp(g/float64(len(qs))), mn, mx)
}

var csPlaceShape = flag.String("cs.place", "512,512", "TestCsPlace2D: shape")

// TestCsPlace2D times one N-D plan on -cs.pairs separately allocated src/dst
// pairs, rotated, to tell a difference between allocations from one between
// plans: per pair, the median ns and the addresses' offsets.
func TestCsPlace2D(t *testing.T) {
	rounds, ms := csRounds(t)
	shape := csInts(t, *csPlaceShape)
	p := NewPlanN(shape...)
	size := shapeProduct(shape...)
	var fns []func()
	var offs []string
	var keep [][]complex128
	for range *csPairs {
		src, dst := benchComplex(size), make([]complex128, size)
		keep = append(keep, src, dst)
		fns = append(fns, func() { p.FFT(dst, src) })
		offs = append(offs, fmt.Sprintf("src%%2M=%7x dst%%2M=%7x", uintptr(unsafe.Pointer(&src[0]))%(2<<20), uintptr(unsafe.Pointer(&dst[0]))%(2<<20)))
	}
	times := comp2Rotate(fns, rounds, ms)
	for k := range fns {
		lo, hi := slices.Min(times[k]), slices.Max(times[k])
		fmt.Printf("PLACE %v pair %d %s median %.0f ns [%.0f %.0f]\n", shape, k, offs[k], median(times[k]), lo, hi)
	}
	_ = keep
}

var csPairs = flag.Int("cs.pairs", 8, "TestCsPlace2D: allocations")

var csReals = flag.String("cs.reals", "65536,131072,1048576,32768", "TestCsUntangleMax: real lengths")

// TestCsUntangleMax times RFFT and IRFFT with the 512-bit untangle and
// retangle bounded at m = 16384 (main) and unbounded, in one process,
// rotated.
func TestCsUntangleMax(t *testing.T) {
	rounds, ms := csRounds(t)
	defer func(a, b int) { kernels.UntangleWideMax, kernels.RetangleWideMax = a, b }(kernels.UntangleWideMax, kernels.RetangleWideMax)
	for _, n := range csInts(t, *csReals) {
		p := NewRealPlan(n)
		src := benchReal(n)
		spec := p.RFFT(make([]complex128, n/2+1), src)
		dst, back := make([]complex128, n/2+1), make([]float64, n)
		for _, inv := range []bool{false, true} {
			run := func(max int) func() {
				return func() {
					kernels.UntangleWideMax, kernels.RetangleWideMax = max, max
					if inv {
						p.IRFFT(back, spec)
					} else {
						p.RFFT(dst, src)
					}
				}
			}
			times := comp2Rotate([]func(){run(1 << 14), run(1 << 30)}, rounds, ms)
			lo, hi := 1e300, 0.0
			for r := range times[0] {
				q := times[0][r] / times[1][r]
				lo, hi = min(lo, q), max(hi, q)
			}
			fmt.Printf("UNT %8d inverse=%v main %10.0f ns  wide %10.0f ns  main/wide %.3f [%.3f %.3f]\n", n, inv,
				median(times[0]), median(times[1]), median(times[0])/median(times[1]), lo, hi)
		}
	}
}

var csExport = flag.String("cs.export", "", "TestCsExport: directory the inputs and outputs go to")

// TestCsExport writes random inputs and the outputs with Round 35's changes
// on ("new") and off ("old": main's cascade orders, no radix 9, the frame
// kernels off, the 512-bit untangle to 16384), as little-endian float64, so
// numpy can judge both on the bytes this code received.
func TestCsExport(t *testing.T) {
	if *csExport == "" {
		t.Skip("set -cs.export=dir")
	}
	r := rand.New(rand.NewPCG(35, 2026))
	write := func(name string, x []float64) {
		b := make([]byte, 0, 8*len(x))
		for _, v := range x {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
		}
		if err := os.WriteFile(filepath.Join(*csExport, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cplx := func(x []complex128) []float64 { return unsafe.Slice((*float64)(unsafe.Pointer(&x[0])), 2*len(x)) }
	old := map[int][]int{}
	for n, f := range splitTable {
		if csCascadeOrders()[n] == nil {
			old[n] = f
		}
	}
	tab, r9, two, three, unt := splitTable, csRadix9On, znTwoPassMax, znThreePassMax, kernels.UntangleWideMax
	defer func() {
		splitTable, csRadix9On, znTwoPassMax, znThreePassMax, kernels.UntangleWideMax = tab, r9, two, three, unt
	}()
	set := func(on bool) {
		splitTable, csRadix9On, znTwoPassMax, znThreePassMax, kernels.UntangleWideMax = old, false, 0, 0, 1<<14
		if on {
			splitTable, csRadix9On, znTwoPassMax, znThreePassMax, kernels.UntangleWideMax = tab, r9, two, three, unt
		}
	}
	tags := map[bool]string{false: "old", true: "new"}
	for _, n := range []int{16, 128, 256, 512, 81, 324, 1080, 1296, 65536, 1 << 20} {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		name := fmt.Sprintf("c_%d", n)
		write("in_"+name+".bin", cplx(x))
		for _, on := range []bool{false, true} {
			set(on)
			p := NewPlan(n)
			write(fmt.Sprintf("%s_%s.bin", tags[on], name), cplx(p.FFT(make([]complex128, n), x)))
			fmt.Printf("EXPORT %s %s %v zn=%v\n", name, tags[on], p.sk.factors(), p.sk.zn != nil)
		}
	}
	for _, n := range []int{256, 512, 1024, 65536, 131072, 1 << 20} {
		x := make([]float64, n)
		for i := range x {
			x[i] = r.NormFloat64()
		}
		name := fmt.Sprintf("r_%d", n)
		write("in_"+name+".bin", x)
		var spec []complex128
		for _, on := range []bool{false, true} {
			set(on)
			znForget(n / 2)
			p := NewRealPlan(n)
			spec = p.RFFT(make([]complex128, n/2+1), x)
			write(fmt.Sprintf("%s_%s.bin", tags[on], name), cplx(spec))
			znForget(n / 2)
		}
		// The inverse of the new spectrum, old and new.
		iname := fmt.Sprintf("ir_%d", n)
		write("in_"+iname+".bin", cplx(spec))
		for _, on := range []bool{false, true} {
			set(on)
			znForget(n / 2)
			p := NewRealPlan(n)
			write(fmt.Sprintf("%s_%s.bin", tags[on], iname), p.IRFFT(make([]float64, n), spec))
			znForget(n / 2)
		}
	}
}
