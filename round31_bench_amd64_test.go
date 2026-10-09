//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 31 (Cascade Lake): where the
// 2-D 1024² transform spends its time, axis by axis, with the row plan as a
// variant.

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
	"time"
	"unsafe"

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

var (
	r31Untangle = flag.String("r31.untangle", "", "TestR31Untangle: rounds,ms")
	r31Reals    = flag.String("r31.reals", "256,512,1024,2048,4096,8192,65536,1048576,1000,1920", "TestR31Untangle: RFFT lengths")
)

// TestR31Untangle times, per RFFT length, the untangle and retangle alone and
// the whole RFFT and IRFFT, with the 512-bit untangle off (Round 30's AVX2
// kernel, the reference) and on, in rotated rounds in one process.
func TestR31Untangle(t *testing.T) {
	if *r31Untangle == "" {
		t.Skip("set -r31.untangle=rounds,ms")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*r31Untangle, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	defer func(v bool) { kernels.UseUntangleAVX512 = v }(kernels.UseUntangleAVX512)
	for _, s := range strings.Split(*r31Reals, ",") {
		var n int
		if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
			t.Fatal(err)
		}
		p := NewRealPlan(n)
		m := n / 2
		src := benchReal(n)
		dst := make([]complex128, m+1)
		back := make([]float64, n)
		Z := make([]complex128, m)
		zin := asComplex(src[:2*m])
		p.half.execute(Z, zin, false)
		p.RFFT(dst, src)
		zb := make([]complex128, m)
		ops := []struct {
			name string
			fn   func()
		}{
			{"untangle", func() { rfftUntangle(dst, Z, p.tw, m) }},
			{"retangle", func() { irfftRetangle(zb, dst, p.tw, m, 0.5/float64(m)) }},
			{"rfft", func() { p.RFFT(dst, src) }},
			{"irfft", func() { p.IRFFT(back, dst) }},
		}
		var b strings.Builder
		for _, op := range ops {
			fns := []func(){
				func() { kernels.UseUntangleAVX512 = false; op.fn() },
				func() { kernels.UseUntangleAVX512 = true; op.fn() },
			}
			times := comp2Rotate(fns, rounds, ms)
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[1][i]
			}
			fmt.Fprintf(&b, " %s %.0f->%.0f %.3f(%.3f-%.3f)", op.name, median(times[0]), median(times[1]),
				median(ratios), slices.Min(ratios), slices.Max(ratios))
		}
		fmt.Printf("U512 %d%s\n", n, b.String())
	}
}

var r31Export = flag.String("r31.export", "", "TestR31Export: directory the inputs and outputs go to")

// TestR31Export writes random inputs and this round's outputs, with Round
// 31's changes on ("new") and off ("old": main's row order, composite rule
// and AVX2 untangle), as little-endian float64, so numpy can judge both on
// the bytes this code received: 2-D 1024² and 512×1024, complex 1000, 1920,
// 2000, 6000, 10000, and RFFT and IRFFT 1024, 4096, 32768.
func TestR31Export(t *testing.T) {
	if *r31Export == "" {
		t.Skip("set -r31.export=dir")
	}
	r := rand.New(rand.NewPCG(31, 2026))
	write := func(name string, x []float64) {
		b := make([]byte, 0, 8*len(x))
		for _, v := range x {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
		}
		if err := os.WriteFile(filepath.Join(*r31Export, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cplx := func(x []complex128) []float64 { return unsafe.Slice((*float64)(unsafe.Pointer(&x[0])), 2*len(x)) }
	randC := func(n int) []complex128 {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		return x
	}
	saved := [3]any{clRowOrder, hswCompOn, kernels.UseUntangleAVX512}
	defer func() {
		clRowOrder, hswCompOn, kernels.UseUntangleAVX512 = saved[0].(map[int][]int), saved[1].(bool), saved[2].(bool)
	}()
	set := func(on bool) {
		clRowOrder, hswCompOn, kernels.UseUntangleAVX512 = nil, false, false
		if on {
			clRowOrder, hswCompOn, kernels.UseUntangleAVX512 = saved[0].(map[int][]int), saved[1].(bool), saved[2].(bool)
		}
	}
	for _, shape := range [][]int{{1024, 1024}, {512, 1024}} {
		x := randC(shape[0] * shape[1])
		name := fmt.Sprintf("nd_%dx%d", shape[0], shape[1])
		write("in_"+name+".bin", cplx(x))
		for _, on := range []bool{false, true} {
			set(on)
			p := NewPlanN(shape...)
			got := p.FFT(make([]complex128, len(x)), x)
			write(fmt.Sprintf("%s_%s.bin", map[bool]string{false: "old", true: "new"}[on], name), cplx(got))
			fmt.Printf("EXPORT %s on=%v rows %v\n", name, on, factorsOf(p.axes[1]))
		}
	}
	for _, n := range []int{1000, 1920, 2000, 6000, 10000} {
		x := randC(n)
		name := fmt.Sprintf("c_%d", n)
		write("in_"+name+".bin", cplx(x))
		for _, on := range []bool{false, true} {
			set(on)
			p := newSKPlanFactors(n, compFactorize(n))
			got := make([]complex128, n)
			p.transform(got, x, false)
			write(fmt.Sprintf("%s_%s.bin", map[bool]string{false: "old", true: "new"}[on], name), cplx(got))
			fmt.Printf("EXPORT %s on=%v factors %v\n", name, on, factorsOf(&Plan{sk: p}))
		}
	}
	for _, n := range []int{1024, 4096, 32768} {
		x := make([]float64, n)
		for i := range x {
			x[i] = r.NormFloat64()
		}
		name := fmt.Sprintf("r_%d", n)
		write("in_"+name+".bin", x)
		spec := randC(n/2 + 1)
		spec[0], spec[n/2] = complex(real(spec[0]), 0), complex(real(spec[n/2]), 0)
		write("in_i"+name+".bin", cplx(spec))
		for _, on := range []bool{false, true} {
			set(on)
			p := NewRealPlan(n)
			tag := map[bool]string{false: "old", true: "new"}[on]
			write(fmt.Sprintf("%s_%s.bin", tag, name), cplx(p.RFFT(make([]complex128, n/2+1), x)))
			write(fmt.Sprintf("%s_i%s.bin", tag, name), p.IRFFT(make([]float64, n), spec))
			fmt.Printf("EXPORT %s on=%v\n", name, on)
		}
	}
}
