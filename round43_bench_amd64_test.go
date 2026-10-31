//go:build amd64

package fft

// Round 43: where complex 1024 spends its time on Zen 3, against FFTW 3.3.10's
// two-pass plan (32·32: n2fv_32 then t2fv_32 in place on the output). Every
// pass of a factorization timed alone on the buffers the transform gives it,
// default and FMA plans, rotated with the whole transform in interleaved
// rounds (comp2Rotate). Skipped unless -z43.ab is set.

import (
	"flag"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/go-fft/fft/internal/kernels"
)

var (
	z43AB   = flag.String("z43.ab", "", "rounds,ms: run TestZ43Passes / TestZ43AB")
	z43Sets = flag.String("z43.sets", "1024:rule", `factorizations, "n:rule,8x8x16"; a "f" suffix (8x8x16f) builds the FMA plan`)
)

// z43Plan is one named plan of a -z43.sets entry.
type z43Plan struct {
	name string
	p    *skPlan
}

// z43Plans parses -z43.sets.
func z43Plans(t *testing.T) (ns []int, sets map[int][]z43Plan) {
	sets = map[int][]z43Plan{}
	for _, spec := range strings.Split(*z43Sets, ";") {
		ns0, vs, ok := strings.Cut(spec, ":")
		if !ok {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(ns0, "%d", &n); err != nil {
			t.Fatal(err)
		}
		for _, v := range strings.Split(vs, ",") {
			fma := strings.HasSuffix(v, "f")
			v = strings.TrimSuffix(v, "f")
			var f []int
			switch v {
			case "rule":
				f = skFactorize(n)
			case "fmarule":
				f, fma = fmaFactors(n), true
			default:
				for _, x := range strings.Split(v, "x") {
					var r int
					if _, err := fmt.Sscanf(x, "%d", &r); err != nil {
						t.Fatal(err)
					}
					f = append(f, r)
				}
			}
			if product(f) != n {
				t.Fatalf("%d: %v multiplies to %d", n, f, product(f))
			}
			name := factorName(f)
			if fma {
				name += "/fma"
			}
			sets[n] = append(sets[n], z43Plan{name, newSKPlanFactorsOpt(n, f, fma)})
		}
		ns = append(ns, n)
	}
	return ns, sets
}

func z43Rounds(t *testing.T) (rounds, ms int) {
	if *z43AB == "" {
		t.Skip("set -z43.ab=rounds,ms")
	}
	if _, err := fmt.Sscanf(*z43AB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	return rounds, ms
}

// TestZ43Passes times every pass of each plan alone, on the buffers and in the
// layout the transform gives it (dst and a scratch buffer off dst's sets, as
// transform takes it), rotated with the whole transform.
func TestZ43Passes(t *testing.T) {
	rounds, ms := z43Rounds(t)
	ns, sets := z43Plans(t)
	for _, n := range ns {
		for _, zp := range sets[n] {
			p := zp.p
			src := benchComplex(n)
			dst := make([]complex128, n)
			bp := p.scratch.Get().(*[]complex128)
			scr := (*bp)[:n]
			if takesGap(n) {
				scr = offTheSets(*bp, dst, n)
			}
			fns := []func(){func() { p.transform(dst, src, false) }}
			s := len(p.stages)
			in := src
			for k := range p.stages {
				out := scr
				if (s-1-k)%2 == 0 {
					out = dst
				}
				st, i0, o0 := &p.stages[k], in, out
				fns = append(fns, func() { st.pass(o0, i0, false) })
				in = out
			}
			times := comp2Rotate(fns, rounds, ms)
			var b strings.Builder
			sum := 0.0
			for k := range p.stages {
				st := &p.stages[k]
				m := median(times[k+1])
				sum += m
				fmt.Fprintf(&b, " r%d/i%d/s%d/f%v %.0f", st.r, st.ido, st.split, st.fma, m)
			}
			fmt.Printf("Z43PASSES %d %-14s whole %.0f sum %.0f |%s\n", n, zp.name, median(times[0]), sum, b.String())
		}
	}
}

// TestZ43Place times each plan's whole transform, and each of its passes
// alone, with src placed at several byte offsets (mod 4 KB) from dst, and
// with the scratch buffer at dst's offset (no gap), all in one arena so the
// offsets are exact. Offset 0 is what two separate 16 KB allocations give.
func TestZ43Place(t *testing.T) {
	rounds, ms := z43Rounds(t)
	ns, sets := z43Plans(t)
	offs := []int{0, 576, 1088, 2048, 2112, 3136}
	for _, n := range ns {
		for _, zp := range sets[n] {
			p := zp.p
			// arena: dst at 0, src at 2n + off/16 points, scratch from the pool.
			arena := make([]complex128, 4*n+512)
			base := 0
			for (uintptr(unsafe.Pointer(&arena[base])) & 4095) != 0 {
				base++
			}
			dst := arena[base : base+n]
			var fns []func()
			var names []string
			for _, off := range offs {
				s0 := base + 2*n + off/16
				src := arena[s0 : s0+n]
				copy(src, benchComplex(n))
				fns = append(fns, func() { p.transform(dst, src, false) })
				names = append(names, fmt.Sprintf("whole@%d", off))
				// the first pass alone, src -> its output buffer
				bp := p.scratch.Get().(*[]complex128)
				scr := (*bp)[:n]
				if takesGap(n) {
					scr = offTheSets(*bp, dst, n)
				}
				out := scr
				if (len(p.stages)-1)%2 == 0 {
					out = dst
				}
				st := &p.stages[0]
				fns = append(fns, func() { st.pass(out, src, false) })
				names = append(names, fmt.Sprintf("pass0@%d", off))
			}
			times := comp2Rotate(fns, rounds, ms)
			var b strings.Builder
			for v := range fns {
				fmt.Fprintf(&b, " %s=%.0f", names[v], median(times[v]))
			}
			fmt.Printf("Z43PLACE %d %-14s%s\n", n, zp.name, b.String())
		}
	}
}

// TestZ43Bal checks the prototype kernel sets (kernels.BalProto) give the
// default kernels' bits, then times the default 1024 plan whole and pass by
// pass with each set, all in rotated rounds in one process.
func TestZ43Bal(t *testing.T) {
	defer kernels.BalProto(0)
	n := 1024
	p := newSKPlan(n)
	src := benchComplex(n)
	for i := range src {
		src[i] = complex(math.Sin(float64(i)*0.37)*1e3, math.Cos(float64(i*i)*0.11)-0.5)
	}
	ref := make([]complex128, n)
	kernels.BalProto(0)
	p.transform(ref, src, false)
	vs := []int{0, 2, 3, 4}
	for _, v := range vs[1:] {
		kernels.BalProto(v)
		got := make([]complex128, n)
		p.transform(got, src, false)
		for i := range got {
			if math.Float64bits(real(got[i])) != math.Float64bits(real(ref[i])) || math.Float64bits(imag(got[i])) != math.Float64bits(imag(ref[i])) {
				t.Fatalf("set %d: bin %d: %v, default %v", v, i, got[i], ref[i])
			}
		}
	}
	kernels.BalProto(0)
	rounds, ms := z43Rounds(t)
	dst := make([]complex128, n)
	bp := p.scratch.Get().(*[]complex128)
	scr := offTheSets(*bp, dst, n)
	bufs := [][2][]complex128{{src, dst}, {dst, scr}, {scr, dst}}
	var fns []func()
	var names []string
	for _, v := range vs {
		fns = append(fns, func() { kernels.BalProto(v); p.transform(dst, src, false) })
		names = append(names, fmt.Sprintf("whole/%d", v))
		for k := range p.stages {
			st, b := &p.stages[k], bufs[k]
			fns = append(fns, func() { kernels.BalProto(v); st.pass(b[1], b[0], false) })
			names = append(names, fmt.Sprintf("p%d/%d", k, v))
		}
	}
	times := comp2Rotate(fns, rounds, ms)
	for q := range fns {
		ratios := make([]float64, rounds)
		base := q % (len(p.stages) + 1) // the same function with set 0
		for i := range ratios {
			ratios[i] = times[base][i] / times[q][i]
		}
		fmt.Printf("Z43BAL %-9s %7.1f ns  default/this %.3f  min %.3f max %.3f\n", names[q], median(times[q]),
			median(ratios), slices.Min(ratios), slices.Max(ratios))
	}
}

// TestZ43AB times each length's plans against the first, whole transforms.
func TestZ43AB(t *testing.T) {
	rounds, ms := z43Rounds(t)
	ns, sets := z43Plans(t)
	for _, n := range ns {
		src := benchComplex(n)
		dst := make([]complex128, n)
		var fns []func()
		for _, zp := range sets[n] {
			p := zp.p
			fns = append(fns, func() { p.transform(dst, src, false) })
		}
		times := comp2Rotate(fns, rounds, ms)
		for v, zp := range sets[n] {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = times[0][i] / times[v][i]
			}
			fmt.Printf("Z43AB %d %-14s %8.1f ns  first/this %.3f  spread %.3f\n", n, zp.name,
				median(times[v]), median(ratios), slices.Max(ratios)/slices.Min(ratios))
		}
	}
}
