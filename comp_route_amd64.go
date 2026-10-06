package fft

import "github.com/go-fft/fft/internal/kernels"

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

// compFactorize is skFactorize past radix16Table: on AMD comp2Factors, on
// Intel a composite's radix-12 passes, else its radix-16 tail; else
// skFactorizeOrder's factorization.
func compFactorize(n int) []int {
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
