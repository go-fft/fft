package fft

// The benchmarks behind BENCHMARKS.md's Round 19: radix-16 passes against
// the current factorization of each power of two.

import (
	"strconv"
	"strings"
	"testing"
)

// radix16Candidates returns the factorizations of 2^e timed against the
// current rule: every distinct ordering of 16^q with the remainder as one
// radix 2, 4 or 8, and of 16^(q-1) with the remaining eight bits or fewer as
// two radix-4/8 passes (32 = 4·8, 64 = 8·8, 128 = 8·16 is already above).
func radix16Candidates(e int) [][]int {
	q, rem := e/4, e%4
	var sets [][]int
	base := func(k int) []int {
		f := make([]int, k)
		for i := range f {
			f[i] = 16
		}
		return f
	}
	if q >= 1 {
		f := base(q)
		if rem > 0 {
			f = append(f, 1<<rem)
		}
		sets = append(sets, f)
		switch rem {
		case 1:
			sets = append(sets, append(base(q-1), 4, 8))
		case 2:
			sets = append(sets, append(base(q-1), 8, 8))
		}
	}
	var out [][]int
	for _, s := range sets {
		out = append(out, perms(s)...)
	}
	return out
}

func factorName(f []int) string {
	var s []string
	for _, r := range f {
		s = append(s, strconv.Itoa(r))
	}
	return strings.Join(s, "x")
}

// BenchmarkRadix16Rules times the current factorization ("cur") and the
// radix-16 candidates of every power of two from 32 to 2^20.
func BenchmarkRadix16Rules(b *testing.B) {
	for e := 5; e <= 20; e++ {
		n := 1 << e
		src := benchComplex(n)
		dst := make([]complex128, n)
		cands := append([][]int{skFactorize(n)}, radix16Candidates(e)...)
		for c, f := range cands {
			p := newSKPlanFactors(n, f)
			name := strconv.Itoa(n) + "/" + factorName(f)
			if c == 0 {
				name = strconv.Itoa(n) + "/cur-" + factorName(f)
			}
			b.Run(name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.transform(dst, src, false)
				}
			})
		}
	}
}

// BenchmarkRadix16Passes times each pass of a factorization alone, on the
// buffers the transform gives it (BenchmarkDecomp for a chosen factor list).
func BenchmarkRadix16Passes(b *testing.B) {
	for _, f := range [][]int{{4, 8, 8}, {16, 16}, {8, 16}, {4, 4, 8}, {4, 4, 8, 8}, {4, 16, 16}, {16, 16, 4}, {16, 16, 16}, {4, 4, 8, 8, 8, 8}} {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
		src := benchComplex(n)
		dst := make([]complex128, n)
		bp := p.scratch.Get().(*[]complex128)
		scr := (*bp)[:n]
		if takesGap(n) {
			scr = offTheSets(*bp, dst, n)
		}
		s := len(p.stages)
		in := src
		prefix := strconv.Itoa(n) + "/" + factorName(f) + "/"
		b.Run(prefix+"whole", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.transform(dst, src, false)
			}
		})
		for k := range p.stages {
			out := scr
			if (s-1-k)%2 == 0 {
				out = dst
			}
			st, i0, o0 := &p.stages[k], in, out
			b.Run(prefix+"pass"+strconv.Itoa(k)+"_r"+strconv.Itoa(st.r)+"_ido"+strconv.Itoa(st.ido), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					st.pass(o0, i0, false)
				}
			})
			in = out
		}
		p.scratch.Put(bp)
	}
}
