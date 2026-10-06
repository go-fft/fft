package fft

// The benchmarks behind BENCHMARKS.md's Round 24: where the time of a smooth
// composite length goes, pass by pass, and the factorizations tried for it.

import (
	"os"
	"strconv"
	"strings"
	"testing"
)

// compR24Sizes are the composite lengths Round 24 decomposes.
var compR24Sizes = []int{1000, 1080, 1296, 1920, 2000, 6000}

// compFactName spells a factorization as its radices joined by '.'.
func compFactName(f []int) string {
	s := make([]string, len(f))
	for i, r := range f {
		s[i] = strconv.Itoa(r)
	}
	return strings.Join(s, ".")
}

// compR24Plans returns the factorizations to time: the default plan of each
// size, then those listed in R24_FACTS ("1000:5.5.5.8;1296:3.3.3.3.16"), whose
// product must be the size.
func compR24Plans() [][]int {
	var out [][]int
	for _, n := range compR24Sizes {
		out = append(out, skFactorize(n))
	}
	for _, spec := range strings.Split(os.Getenv("R24_FACTS"), ";") {
		_, rs, ok := strings.Cut(spec, ":")
		if !ok {
			continue
		}
		var f []int
		for _, s := range strings.Split(rs, ".") {
			r, err := strconv.Atoi(s)
			if err != nil {
				panic(err)
			}
			f = append(f, r)
		}
		out = append(out, f)
	}
	return out
}

// BenchmarkR24Decomp times each listed factorization whole (out of place, as
// the parity harness calls it) and each of its passes alone, on the buffers
// the transform gives that pass.
func BenchmarkR24Decomp(b *testing.B) {
	compR24Warm(b)
	for _, f := range compR24Plans() {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
		src := benchComplex(n)
		dst := make([]complex128, n)
		name := strconv.Itoa(n) + "-" + compFactName(f)
		b.Run(name+"-whole", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.transform(dst, src, false)
			}
		})
		bp := p.scratch.Get().(*[]complex128)
		scr := (*bp)[:n]
		s := len(p.stages)
		in := src
		for k := range p.stages {
			out := scr
			if (s-1-k)%2 == 0 {
				out = dst
			}
			st, i0, o0 := &p.stages[k], in, out
			b.Run(name+"-p"+strconv.Itoa(k)+"r"+strconv.Itoa(st.r)+"i"+strconv.Itoa(st.ido), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					st.pass(o0, i0, false)
				}
			})
			in = out
		}
		p.scratch.Put(bp)
	}
}

// compR24Warm runs a transform for about a second before anything is timed,
// so the first row does not pay for the core's clock ramp.
func compR24Warm(b *testing.B) {
	b.Run("0warm", func(b *testing.B) {
		p := NewPlan(1000)
		src := benchComplex(1000)
		dst := make([]complex128, 1000)
		for i := 0; i < b.N; i++ {
			p.FFT(dst, src)
		}
	})
}

// compTails returns every ordered sequence of radices 2, 4, 8 and 16 whose
// product is 2^e, with at most one radix 2 and at most four passes.
func compTails(e int) [][]int {
	var out [][]int
	var rec func(left int, cur []int, twos int)
	rec = func(left int, cur []int, twos int) {
		if left == 0 {
			out = append(out, append([]int(nil), cur...))
			return
		}
		if len(cur) == 4 {
			return
		}
		for _, b := range []int{1, 2, 3, 4} {
			if b > left || (b == 1 && twos > 0) {
				continue
			}
			t := twos
			if b == 1 {
				t++
			}
			rec(left-b, append(cur, 1<<b), t)
		}
	}
	rec(e, nil, 0)
	return out
}

// compR24TailSizes are the composites the tail sweep times: an odd part made
// of 3s and 5s times 2^e.
var compR24TailSizes = []int{
	240, 400, 720, 1296, 2000, 6000, // e = 4
	480, 800, 1440, 4000, // 5
	576, 960, 1600, 2880, 8000, // 6
	384, 640, 1920, 3200, // 7
	768, 1280, 3840, 6400, // 8
	1536, 2560, 7680, // 9
	3072, 5120, 15360, // 10
}

// BenchmarkR24Tails times, for each length of compR24TailSizes, its odd
// radices first (3s, then 5s) followed by every power-of-two tail compTails
// gives. R24_TAILN restricts it to the listed lengths ("1296,2000").
func BenchmarkR24Tails(b *testing.B) {
	compR24Warm(b)
	sizes := compR24TailSizes
	if s := os.Getenv("R24_TAILN"); s != "" {
		sizes = nil
		for _, f := range strings.Split(s, ",") {
			n, err := strconv.Atoi(f)
			if err != nil {
				panic(err)
			}
			sizes = append(sizes, n)
		}
	}
	for _, n := range sizes {
		e, odd := 0, n
		for odd%2 == 0 {
			odd /= 2
			e++
		}
		var o []int
		for _, q := range []int{3, 5} {
			for odd%q == 0 {
				o = append(o, q)
				odd /= q
			}
		}
		if odd != 1 {
			panic("compR24TailSizes: not 2^e·3^a·5^b")
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		for _, t := range compTails(e) {
			f := append(append([]int(nil), o...), t...)
			p := newSKPlanFactors(n, f)
			b.Run(strconv.Itoa(n)+"-"+compFactName(t), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.transform(dst, src, false)
				}
			})
		}
	}
}

// BenchmarkR24AB is the row set Round 24's interleaved A/B runs time: the
// composites, the lengths with a factor 7, powers of two, the prime engines
// (Rader 1201 has a 2^4·3·5^2 convolution, 1009 a 2^4·3^2·7 one), real
// transforms and 2-D plans of composite sides.
func BenchmarkR24AB(b *testing.B) {
	compR24Warm(b)
	for _, n := range []int{1000, 1080, 1296, 1920, 2000, 6000, 240, 480, 720, 768, 960, 3072, 3840, 15360, 45000,
		1008, 2100, 20160, 256, 1024, 4096, 65536, 1009, 1201, 10007} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C-"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
	for _, n := range []int{1000, 1920, 2000, 4096} {
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		b.Run("R-"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, src)
			}
		})
	}
	for _, n := range []int{100, 120, 240} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		b.Run("2D-"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
}

// compR24Rule12 is a candidate factorization with radix-12 passes for n =
// 2^e·3^a·5^b (a >= 1, e >= 2): t = min(a, e/2) radix-12 passes last, the
// remaining 3s and 5s first, then the remaining powers of two (radix8Maximal,
// radix 4 before 8). nil for any other n.
func compR24Rule12(n int) []int {
	e, odd := 0, n
	for odd%2 == 0 {
		odd /= 2
		e++
	}
	a := 0
	for odd%3 == 0 {
		odd /= 3
		a++
	}
	b := 0
	for odd%5 == 0 {
		odd /= 5
		b++
	}
	if odd != 1 || a == 0 || e < 2 {
		return nil
	}
	t := min(a, e/2)
	var f []int
	for range a - t {
		f = append(f, 3)
	}
	for range b {
		f = append(f, 5)
	}
	tail := radix8Maximal(e - 2*t)
	for i := len(tail) - 1; i >= 0; i-- {
		f = append(f, tail[i])
	}
	for range t {
		f = append(f, 12)
	}
	return f
}

// BenchmarkR24Rule12 times, for every n = 2^e·3^a·5^b up to 16384 with a >= 1
// and e >= 2, skFactorize's factorization against compR24Rule12's.
func BenchmarkR24Rule12(b *testing.B) {
	compR24Warm(b)
	for n := 12; n <= 1<<14; n++ {
		g := compR24Rule12(n)
		if g == nil {
			continue
		}
		src := benchComplex(n)
		dst := make([]complex128, n)
		for _, f := range [][]int{skFactorize(n), g} {
			p := newSKPlanFactors(n, f)
			b.Run(strconv.Itoa(n)+"-"+compFactName(f), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.transform(dst, src, false)
				}
			})
		}
	}
}
