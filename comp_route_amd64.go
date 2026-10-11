package fft

import (
	"slices"

	"github.com/go-fft/fft/internal/kernels"
)

// compOddFirstDefault is oddRadicesFirst on amd64 (Round 17's order with the
// AVX2 kernels).
func compOddFirstDefault() bool { return oddRadicesFirstDefault() }

// compRadix16On enables compRadix16For and compRadix12For: the AVX2 kernels on
// Intel. A
// variable so the tests can take both routes on any machine.
var compRadix16On = kernels.UseStockhamAVX2 && kernels.IntelCPU

// compRadix16 is compRadix16For on this machine.
func compRadix16(n int) []int { return compRadix16For(n, compRadix16On) }

// comp2On enables comp2Factors: the AVX2 kernels on a CPU that is not
// Intel's (Round 26 measured Zen 3). A variable so the tests can take both
// routes on any machine.
var comp2On = kernels.UseStockhamAVX2 && !kernels.IntelCPU

// hswCompOn enables hswComp2Factors: the AVX2 kernels on an Intel CPU,
// with or without the AVX-512 ones (Haswell, Round 29; Cascade Lake, Round
// 31, where a composite never runs the AVX-512 kernels: over the 150 lengths
// 2^e·3^a·5^b up to 16384, five rotated rounds, it ran 1.086× as fast as
// Round 24's Intel rule interleaved, 1.093× with the 256-bit split layout,
// kernels.ClCompSplit256). A variable so the tests can take the route on any
// machine.
var hswCompOn = kernels.UseStockhamAVX2 && kernels.IntelCPU

// compFactorize is skFactorize past radix16Table: on Haswell, hswE11Factors;
// on AMD zmPow2Tail, else comp2Factors, on Intel hswComp2Factors; where those
// give none, on Intel a composite's
// radix-12 passes, else its radix-16 tail; else skFactorizeOrder's
// factorization.
func compFactorize(n int) []int { return csRadix9For(csCompFactorize(n), csRadix9On) }

// csCompFactorize is compFactorize before csRadix9For.
func csCompFactorize(n int) []int {
	if f := hswE11Factors(n, hswE11On); f != nil {
		return f
	}
	if f := hswComp2Factors(n, hswCompOn); f != nil {
		return f
	}
	if f := zmPow2Tail(n, comp2On); f != nil {
		return f
	}
	if f := comp2Factors(n, comp2On); f != nil {
		return f
	}
	if f := compRadix12For(n, compRadix16On); f != nil {
		return f
	}
	if f := compRadix16(n); f != nil {
		return f
	}
	return skFactorizeOrder(n, compOddFirst)
}

// compRadix12For gives the factorization of n = 2^e·3^a·5^b (a >= 1, n <=
// 16384) with t = min(a, ⌊e/2⌋) radix-12 passes (radix12.go), or nil when n
// keeps its other factorization: the remaining 3s and 5s first, then the
// remaining powers of two as radix8Maximal (radix 4 before 8), then the
// radix-12 passes, the last of them the final pass. Each radix-12 pass takes
// a 3 and a 4: on Cascade Lake (Round 24, BenchmarkR24Rule12, three rounds,
// one pinned core, all 96 such lengths up to 16384 with e >= 2) it ran
// 1.03–1.40× as fast as the factorization it replaces for 2 <= e <= 8 except
// e = 3, and for e = 9 when t >= 2 (1.17–1.28). With e = 3, or e = 9 and t =
// 1, it tied (0.97–1.02), and from e = 10 on it lost to the radix-16 tail
// (0.78–0.97), so those keep it.
func compRadix12For(n int, on bool) []int {
	if !on || n > 1<<14 {
		return nil
	}
	e, odd := 0, n
	for odd%2 == 0 {
		odd /= 2
		e++
	}
	a, b := 0, 0
	for ; odd%3 == 0; odd /= 3 {
		a++
	}
	for ; odd%5 == 0; odd /= 5 {
		b++
	}
	t := min(a, e/2)
	if odd != 1 || t == 0 || e >= 10 || t == 1 && (e == 3 || e == 9) {
		return nil
	}
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

// compRadix16For gives the factorization of a composite n = 2^e·3^a·5^b (a+b
// >= 1, n <= 16384) whose power-of-two part ends with radix-16 passes, or nil
// when n keeps skFactorizeOrder's: the odd radices first, then
//
//	e = 4, 7, 10:  8^((e-4)/3) · 16
//	e = 8:         16 · 16
//
// and every other e as before. A composite never runs the AVX-512 kernels
// (wide512), so the AVX2 radix-16 kernels serve Cascade Lake too, which the
// powers of two of radix16TableAMD64 leave out. Every power-of-two tail of
// radices 2, 4, 8 and 16 (at most four passes, one radix 2) was timed after
// the odd passes of 29 lengths from 240 to 15360 on Cascade Lake (Round 24,
// one pinned core, three rounds, BenchmarkR24Tails). The current tail's time
// ÷ the best one's: e = 4 1.08–1.15 (best 16), e = 7 1.07–1.11 (8·16 and 16·8
// within 0.3%), e = 8 1.08–1.10 (16·16), e = 10 1.17–1.23 (8·8·16). For e = 5,
// 6 and 9 the current tail was best or within 1.4% of it. The final radix-16
// pass of a composite writes sixteen streams l1 = n/16 points apart, never a
// multiple of 4 KB, so Round 19's set conflict does not arise below 16384.
// Only Intel runs it: AMD was not measured (Round 19 found the vendors split
// on radix 16 at 2048).
func compRadix16For(n int, on bool) []int {
	if !on || n > 1<<14 {
		return nil
	}
	e, odd := 0, n
	for odd%2 == 0 {
		odd /= 2
		e++
	}
	var f []int
	for _, q := range []int{3, 5} {
		for odd%q == 0 {
			f = append(f, q)
			odd /= q
		}
	}
	if odd != 1 || len(f) == 0 {
		return nil
	}
	switch {
	case e == 8:
		return append(f, 16, 16)
	case e >= 4 && e%3 == 1:
		for ; e > 4; e -= 3 {
			f = append(f, 8)
		}
		return append(f, 16)
	}
	return nil
}

// comp2Factors gives the factorization of n = 2^e·3^a·5^b (e >= 1, a+b >= 1,
// n <= 16384) with prime-factor passes of radix 12 (radix12.go) and 10, 15,
// 20 (radix5q.go), or nil when n keeps skFactorizeOrder's (or on is false):
//
//   - t12 = min(a, ⌊e/2⌋) radix-12 passes, then t20 = min(b, ⌊e'/2⌋) radix-20
//     passes on the powers of two left, then one radix-10 pass if a single 2
//     and a 5 are left; a single 2 left over otherwise goes back with the last
//     radix-20 (else radix-12) pass, so the tail is a radix-8 pass and not a
//     radix-2 one;
//   - one radix-15 pass if a 3 and a 5 are both left;
//   - the order: the remaining 3s and 5s, the remaining powers of two
//     (radix8Maximal, radix 4 before 8), then the 15, 10, 12 and 20 passes,
//     the last of them the final pass.
//
// Each of those passes writes r streams n/r points apart, which fall into
// few L1 sets when n/r carries a high power of two: radix 12 is used for e
// <= 9, radix 20 for e <= 8, radix 10 for e <= 7 and radix 15 for e <= 6,
// which keeps those strides off a multiple of 2 KB (4 KB for radix 12). With
// e = 10 the power of two ends with 8·8·16, as Round 24's compRadix16For
// gives on Intel; with e >= 11 n keeps its factorization.
//
// Every factorization of these radices (comp2Candidates, round26 tests) was
// timed for all 148 such lengths up to 16384 on Zen 3 (Round 26, one pinned
// core, five interleaved rounds): this rule's choice ran within 0.7% of the
// best candidate (geometric mean), never slower than the factorization it
// replaces, and 1.10× as fast as it (geometric mean).
func comp2Factors(n int, on bool) []int {
	if !on || n > 1<<14 {
		return nil
	}
	e, odd := 0, n
	for odd%2 == 0 {
		odd /= 2
		e++
	}
	a, b := 0, 0
	for ; odd%3 == 0; odd /= 3 {
		a++
	}
	for ; odd%5 == 0; odd /= 5 {
		b++
	}
	if odd != 1 || e == 0 || a+b == 0 || e >= 11 {
		return nil
	}
	rep := func(f []int, r, c int) []int {
		for range c {
			f = append(f, r)
		}
		return f
	}
	if e == 10 {
		return append(rep(rep(nil, 3, a), 5, b), 8, 8, 16)
	}
	var t12, t20, t10, t15 int
	t12 = min(a, e/2)
	if e <= 8 {
		t20 = min(b, (e-2*t12)/2)
	}
	ep := e - 2*t12 - 2*t20
	if e <= 7 && ep%2 == 1 && b > t20 {
		t10 = 1
	}
	ep -= t10
	switch {
	case ep == 1 && t20 > 0:
		t20--
		ep += 2
	case ep == 1 && t12 > 0 && e >= 3:
		t12--
		ep += 2
	}
	if e <= 6 && a > t12 && b > t20+t10 {
		t15 = 1
	}
	f := rep(rep(nil, 3, a-t12-t15), 5, b-t20-t10-t15)
	tail := radix8Maximal(ep)
	for i := len(tail) - 1; i >= 0; i-- {
		f = append(f, tail[i])
	}
	return rep(rep(rep(rep(f, 15, t15), 10, t10), 12, t12), 20, t20)
}

// hswComp2Factors is comp2Factors with Round 24's radix-16 tails: where
// comp2Factors closes the powers of two left over by its radix-12 and -20
// passes with 4·4 (four bits) or 4·4·8 (seven), it takes 16 or 8·16 instead.
// Every factorization comp2Candidates lists was timed for the 145 lengths n
// = 2^e·3^a·5^b up to 16384 (e >= 1, a+b >= 1) on Haswell (Round 29, one
// pinned core, five rotated rounds). comp2Factors' choice took 1.013× the
// best candidate's time (geometric mean) and lost to Round 24's Intel rule
// at 1536, 3840, 6400 and 7680 (0.945–0.990), all of them a 4·4 or 4·4·8
// tail; with the radix-16 tails, 1.009× the best, 1.077× as fast as Round
// 24's rule, none slower than it beyond 1.6%.
func hswComp2Factors(n int, on bool) []int {
	f := comp2Factors(n, on)
	var tail []int
	at := -1
	for k, r := range f {
		if r == 4 || r == 8 {
			if at < 0 {
				at = k
			}
			tail = append(tail, r)
		}
	}
	var r16 []int
	switch {
	case slices.Equal(tail, []int{4, 4}):
		r16 = []int{16}
	case slices.Equal(tail, []int{4, 4, 8}):
		r16 = []int{8, 16}
	default:
		return f
	}
	return slices.Concat(f[:at], r16, f[at+len(tail):])
}

// hswE11On enables hswE11Factors: the AVX2 kernels on an Intel CPU without
// the AVX-512 ones (Haswell, Round 37; Cascade Lake was not measured). A
// variable so the tests can take the route on any machine.
var hswE11On = hswE11Default(kernels.UseStockhamAVX2, kernels.IntelCPU, kernels.UseStockhamAVX512)

// hswE11Default is hswE11On's choice: AVX2 on Intel without AVX-512.
func hswE11Default(avx2, intel, avx512 bool) bool { return avx2 && intel && !avx512 }

// hswE11Factors gives n = 2^11·p, p = 3 or 5 (6144 and 10240, the lengths
// 2^e·3^a·5^b up to 16384 with e = 11 and one odd factor), the factorization
// p·8·16·16, or nil. No rule covered e >= 11 (comp2Factors stops at e = 10),
// so they took 3·4·8·8·8 and 5·4·8·8·8. On Haswell (Round 37, one pinned
// core) every factorization of the 3 or 5, the radix-10, -12 and -20 passes
// and every ordering of radix-4, -8 and -16 tails was timed (5 rotated
// rounds, 35 and 52 candidates): p·8·16·16 was the best of both, 1.173× and
// 1.211× as fast as the old one, every round within 0.003 of that; 12288
// (e = 12, 3·8^4) had none better than 1.010, so it keeps its own.
func hswE11Factors(n int, on bool) []int {
	if !on || n&(1<<11-1) != 0 {
		return nil
	}
	switch p := n >> 11; p {
	case 3, 5:
		return []int{p, 8, 16, 16}
	}
	return nil
}

// csRadix9On enables csRadix9For: the AVX2 kernels on an Intel CPU with
// AVX-512 (Cascade Lake, Round 35, where it was measured). A variable so the
// tests can take the route on any machine.
var csRadix9On = kernels.UseStockhamAVX2 && kernels.UseStockhamAVX512 && kernels.IntelCPU

// csRadix9For merges the leading radix-3 passes of a factorization two by two
// into radix-9 passes (radix9.go), first, a lone 3 after them; f is returned
// unchanged when on is false or it opens with fewer than two radix-3 passes.
func csRadix9For(f []int, on bool) []int {
	c := 0
	for c < len(f) && f[c] == 3 {
		c++
	}
	if !on || c < 2 {
		return f
	}
	var g []int
	for range c / 2 {
		g = append(g, 9)
	}
	if c%2 == 1 {
		g = append(g, 3)
	}
	return append(g, f[c:]...)
}

// zmPow2Tail gives the factorization of n = 2^e·3^a·5^b (a+b >= 1) with e =
// 11 or 12 and n <= 32768, or nil (or when on is false): the 3s and 5s
// first, in skFactorizeOrder's order, then 2^e as radix-4 passes with, for e
// = 11, one radix-8 pass before the final radix-4 one: 4·4·4·8·4 (e = 11),
// 4·4·4·4·4·4 (e = 12). comp2Factors leaves e >= 11 alone, and
// skFactorizeOrder ends such a length with radix8Maximal (6144 = 3·4·8·8·8).
// A final radix-8 pass writes eight streams n/8 points apart, 2^(e+1)·odd
// bytes, a multiple of 4 KB from e = 11 on, so all eight fall in one L1 set:
// at 6144 points on Zen 3 it took 4.6 µs of 18.8, the final radix-4 pass
// that replaces it 2.0 (Round 36, TestR26Passes). Every ordered tail of
// radices 4, 8 and 16 was timed after the odd passes of 6144, 10240, 12288,
// 18432, 20480 and 30720 (TestZmTails, nine rotated rounds); this one was
// the best or within 1.5% of it at 6144, 10240, 12288 and 18432. Against
// the factorization it replaces (TestR26AB, fifteen rotated rounds) it ran
// 1.09× as fast at 6144, 1.07× at 10240, 1.04× at 18432, 1.02× at 30720,
// 1.08× at 12288 and 1.06× at 20480 (Bluestein's convolution for 10007).
// From 36864 points and at e >= 13 no tail gained consistently
// (TestZmPow2Rule), so those keep their factorization; Intel was not
// measured.
func zmPow2Tail(n int, on bool) []int {
	if !on || n > 1<<15 {
		return nil
	}
	e, odd := 0, n
	for odd%2 == 0 {
		odd /= 2
		e++
	}
	if e < 11 || e > 12 || odd == 1 {
		return nil
	}
	var f []int
	for _, q := range []int{3, 5} {
		for odd%q == 0 {
			f = append(f, q)
			odd /= q
		}
	}
	if odd != 1 {
		return nil
	}
	f = append(f, 4, 4, 4)
	if e == 11 {
		return append(f, 8, 4)
	}
	return append(f, 4, 4, 4)
}
