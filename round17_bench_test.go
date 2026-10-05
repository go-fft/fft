package fft

// The benchmarks behind BENCHMARKS.md's Round 17: where a transform's time
// goes, the radix orders, the strip widths and the fan-out floor, and the row
// set the interleaved A/B runs time.

import (
	"strconv"
	"testing"
)

// BenchmarkDecomp splits a Stockham transform into its parts: the whole call,
// each pass alone on the buffers the transform gives it, and the call with
// the passes removed (pool, scratch placement, dispatch).
func BenchmarkDecomp(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512, 1024, 4096, 1000} {
		p := NewPlan(n)
		if p.sk == nil {
			continue
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n)+"/whole", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
		sk := p.sk
		bp := sk.scratch.Get().(*[]complex128)
		scr := (*bp)[:n]
		if takesGap(n) {
			scr = offTheSets(*bp, dst, n)
		}
		s := len(sk.stages)
		in := src
		for k := range sk.stages {
			out := scr
			if (s-1-k)%2 == 0 {
				out = dst
			}
			st, i0, o0 := &sk.stages[k], in, out
			b.Run(strconv.Itoa(n)+"/pass"+strconv.Itoa(k)+"_r"+strconv.Itoa(st.r)+"_ido"+strconv.Itoa(st.ido), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					st.pass(o0, i0, false)
				}
			})
			in = out
		}
		sk.scratch.Put(bp)
		b.Run(strconv.Itoa(n)+"/nopass", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				bp := sk.scratch.Get().(*[]complex128)
				scr := (*bp)[:n]
				if takesGap(n) {
					scr = offTheSets(*bp, dst, n)
				}
				_ = scr
				p.checkLen(dst, src)
				sk.scratch.Put(bp)
			}
		})
	}
}

// BenchmarkDecomp2D times a 2-D plan's parts on one goroutine: the rows (the
// contiguous axis), the columns (gathered), and the whole.
func BenchmarkDecomp2D(b *testing.B) {
	for _, n := range []int{64, 128, 256} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		copy(dst, src)
		name := strconv.Itoa(n) + "x" + strconv.Itoa(n)
		b.Run(name+"/whole", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
		b.Run(name+"/rows", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.contiguousLines(dst, dst, 1, 0, n, false)
			}
		})
		blocks := n / blockWidth(n)
		b.Run(name+"/cols", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.blockedLines(dst, dst, 0, 0, blocks, false)
			}
		})
		if p.strips[0] != nil {
			strips := (n + stripWidth(n) - 1) / stripWidth(n)
			b.Run(name+"/strips", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.stripLines(dst, dst, 0, 0, strips, false)
				}
			})
		}
		b.Run(name+"/copy", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				copy(dst, src)
			}
		})
	}
}

// perms returns the distinct orderings of f.
func perms(f []int) [][]int {
	if len(f) <= 1 {
		return [][]int{append([]int(nil), f...)}
	}
	var out [][]int
	seen := map[int]bool{}
	for i, v := range f {
		if seen[v] {
			continue
		}
		seen[v] = true
		rest := append(append([]int(nil), f[:i]...), f[i+1:]...)
		for _, p := range perms(rest) {
			out = append(out, append([]int{v}, p...))
		}
	}
	return out
}

// BenchmarkOrders times every ordering of each length's radices.
func BenchmarkOrders(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512, 1024, 2048, 1000, 1080, 1296, 1920, 2000, 6000} {
		src := benchComplex(n)
		dst := make([]complex128, n)
		for _, f := range perms(skFactorize(n)) {
			p := newSKPlanFactors(n, f)
			name := strconv.Itoa(n) + "/"
			for _, r := range f {
				name += strconv.Itoa(r)
			}
			b.Run(name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.transform(dst, src, false)
				}
			})
		}
	}
}

// BenchmarkAB is the row set the A/B runs time: complex, real and 2-D plans
// writing into a reused slice.
func BenchmarkAB(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 1 << 18, 1000, 1080, 1296, 1920, 2000, 6000, 1009, 10007} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
	for _, n := range []int{256, 1024, 4096, 1000} {
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		b.Run("R/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, src)
			}
		})
	}
	for _, n := range []int{32, 64, 128, 256, 512, 1024} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		b.Run("2D/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
}

// BenchmarkFanout times 2-D plans on all cores for several parMinChunk values.
func BenchmarkFanout(b *testing.B) {
	defer func(v int) { parMinChunk = v }(parMinChunk)
	for _, n := range []int{64, 128, 256, 512} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		for _, m := range []int{8192, 16384, 32768, 65536} {
			parMinChunk = m
			b.Run(strconv.Itoa(n)+"/m"+strconv.Itoa(m), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.FFT(dst, src)
				}
			})
		}
	}
}
