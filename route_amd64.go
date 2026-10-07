package fft

import (
	"math"

	"github.com/go-fft/fft/internal/kernels"
)

// pow2StockhamMaxDefault routes every power of two to the Stockham engine on
// amd64 when its AVX2 (or AVX-512) pass kernels run.
//
// Until 2026-10-04 the AVX2 route stopped at 4096: measured then (pow2 kernel
// time ÷ Stockham time, 2026-09-30), Haswell and Cascade Lake lost above it
// and Zen 3 lost AT 4096. Those Stockham times were inflated by L1 set
// conflicts between dst and the scratch buffer (see offTheSets); with the
// scratch placed off dst's sets, the same comparison reads (2026-10-04):
//
//	n        4096  8192  16384  32768  65536  2^17  2^18  2^19  2^20
//	Zen 3    1.50  1.58  1.94   1.50   1.81   1.38  1.81  1.96  2.51
//	Haswell  1.76  1.05  1.65   1.04   1.16   0.82  1.15  1.23  1.24
//
// Stockham wins at every size on Zen 3, and on Haswell everywhere but 2^17;
// the geometric mean of the two is above 1 at every size. Without AVX2 the
// Stockham passes run scalar, and lost to the pow2 kernel's SSE2 butterflies
// at every size when measured (2026-09-30, before the scratch fix; not
// re-measured, since every host at hand has AVX2).
func pow2StockhamMaxDefault() int {
	return pow2StockhamMaxAMD64(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
}

// With AVX-512 the route stopped at 16384 (Cascade Lake, pow2 kernel time ÷
// Stockham time, 2026-10-01: 2.33 at 4096, 0.86–1.23 above 16384). Above 4096
// those powers of two were factored with rule B. Radix 8 as far as it goes
// (rule A, see r8MaxPow2AMD64) is 10–38% faster at every size there, and with
// it and the scratch off dst's sets the Stockham engine
// beats the pow2 kernel at every size on the same idle host (2026-10-04,
// ns per point):
//
//	n            32768  65536  2^17   2^18   2^19   2^20
//	Stockham A    6.18   9.00  10.50  12.72  25.49  28.67
//	pow2 kernel   9.91  13.18  15.30  24.41  33.31  36.66
func pow2StockhamMaxAMD64(avx2, avx512 bool) int {
	switch {
	case avx2 || avx512:
		return math.MaxInt
	}
	return 0
}

// r8MaxPow2Default keeps radix-8 passes for every power of two with the
// AVX-512 passes, and with the AVX2 passes on Intel; on other AVX2 CPUs, up to
// 4096.
func r8MaxPow2Default() int {
	return r8MaxPow2AMD64(kernels.UseStockhamAVX512, kernels.IntelCPU)
}

// r8MaxPow2AMD64 is r8MaxPow2Default's choice. Three rules were timed at every
// 2^e from 256 to 2^20: A, radix 8 as far as it goes; B, radix 4 with a
// radix-2 pass for an odd exponent; C, radix 4 with one radix-8 pass for an
// odd exponent. Time over the best of the three, geometric mean (worst), on
// v0.1.6 with the grouped twiddles (2026-10-04; Cascade Lake from v0.1.3):
//
//	                    A to 4096, B   A to 1024, B   A everywhere
//	Haswell (AVX2)      1.173 (1.47)   1.218 (1.47)   1.003 (1.04)
//	Zen 3 (AVX2)        1.017 (1.20)   1.003 (1.03)   1.101 (1.55)
//	Cascade (AVX-512)   1.239 (1.62)   1.290 (1.62)   1.000 (1.00)
//
// The vendors disagree from 4096 up: radix 8 wins every size on Haswell (B
// takes 1.18–1.47× its time) and loses or ties on Zen 3 (1.00–1.50×). One
// rule for both costs one of them 10–17%, so the rule follows the vendor:
// Intel takes radix 8 everywhere, like AVX-512. Every other vendor keeps A to
// 4096 then B, the rule all AVX2 CPUs had: Zen 3 is the only non-Intel CPU
// measured, and it is within 1.4% of its best there, so nothing justifies
// changing what Zen 2, 4 or 5 get on its evidence alone.
func r8MaxPow2AMD64(avx512, intel bool) int {
	if avx512 || intel {
		return math.MaxInt
	}
	return 4096
}

// pow2OneRadix8Max is 0 on amd64: above r8MaxPow2 a power of two takes rule B
// (see r8MaxPow2AMD64). Below it, timed at 2^5..2^13 on Haswell, Zen 3 and
// Cascade Lake (2026-09-30), rule A scored 1.082 over the best, C 1.129, B
// 1.190.
const pow2OneRadix8Max = 0

// oddRadicesFirstDefault orders the passes 3s and 5s first, then the powers
// of two with radix 4 before radix 8 (so the twiddle-free final pass is a
// radix-8 one), then 7, 11, 13, when the AVX2 pass kernels run. Every
// ordering of the radices of twelve lengths was timed (2026-10-05, time over
// the best ordering, geometric mean): the pocketfft order scored 1.063 on Zen
// 3 and 1.082 on Cascade Lake (1000: 1.108 and 1.164), this one 1.008 and
// 1.013. End to end, composites gained 1.08–1.17× and powers of two moved
// within the noise. Radix 7 went
// last after Rader 1009, whose convolution length is 1008 = 2^4·3^2·7, lost
// 5% with 7 among the first passes (it has no SIMD pass).
func oddRadicesFirstDefault() bool { return kernels.UseStockhamAVX2 }

// parMinChunkDefault is 16384 elements when the AVX2 kernels run, 8192
// otherwise. With the batched column passes (PlanN's strips), 128×128 is
// faster on one goroutine than on two: 58 against 115 µs on Zen 3 (128
// threads) and 87 against 101 µs on Cascade Lake (8), all cores, 2026-10-05.
// From 256×256 the two floors were within the noise of each other.
func parMinChunkDefault() int { return parMinChunkAMD64(kernels.UseStockhamAVX2) }

func parMinChunkAMD64(avx2 bool) int {
	if avx2 {
		return 1 << 14
	}
	return 1 << 13
}

// radix16TableDefault is radix16TableAMD64 for this machine.
func radix16TableDefault() map[int][]int {
	return radix16TableAMD64(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512, kernels.IntelCPU)
}

// radix16TableAMD64 gives the powers of two that take radix-16 passes, with
// their factorization, when the AVX2 kernels run and the AVX-512 ones do not.
// Every factorization of 16^q times a radix 2, 4 or 8, or 16^(q-1) times two
// radix-4/8 passes, in every order, was timed against the current rule from
// 32 to 4096 points (Round 19, one core; Zen 3 load 3–4, seven rounds;
// Haswell load 1–2, five rounds). Current time ÷ best common candidate:
//
//	n        128 (16·8)  256 (16·16)  512   1024 (8·8·16)  2048 (8·16·16)  4096
//	Zen 3    1.08        1.05         none  1.05           0.77            0.53
//	Haswell  1.14        1.09         none  1.20           1.22            0.80
//
// 32, 64 and 512 have no radix-16 factorization that wins on both, and from
// 4096 the sixteen output streams of a pass, n/16 points apart, fall into
// one L1 set (Round 8's conflict, now with more streams than ways). 2048
// goes to radix 16 on Intel only, as the radix-8 rule does (r8MaxPow2AMD64).
// AVX-512 machines keep their rule: there is no 512-bit radix-16 kernel, and
// a transform that mixes widths lost in Round 6; except 128 on Intel
// (intelRadix16Table512), where the AVX-512 kernels do not run.
func radix16TableAMD64(avx2, avx512, intel bool) map[int][]int {
	switch {
	case avx2 && avx512 && intel:
		return intelRadix16Table512()
	case !avx2 || avx512:
		return nil
	}
	t := map[int][]int{128: {16, 8}, 256: {16, 16}, 1024: {8, 8, 16}}
	if intel {
		t[2048] = []int{8, 16, 16}
	}
	return t
}

// parThresholdNDefault is parThreshold: the change of Round 21 was measured
// on arm64 only.
func parThresholdNDefault() int { return parThreshold }

// cascadeMinDefault is the smallest power of two that runs the blocked
// schedule (cascade.go): 65536 with the AVX-512 kernels on Intel, never
// otherwise. On Cascade Lake (2026-10-05, main time ÷ new time, fifteen
// interleaved rounds, one core) it gained 1.29 at 65536, 1.36 at 2^17, 1.21
// at 2^18 and 1.24 at 2^20 (2^19: 1.04, within that row's spread); 32768 did
// not move (1.00), so it stays breadth first. With the AVX2 kernels the
// schedule tied breadth first there, because the groups it keeps in L2 are
// then bound by the AVX2 passes' arithmetic. Haswell (Intel, AVX2, 256 KB of
// L2) was not measured, so it keeps breadth first; so does every AMD CPU.
func cascadeMinDefault() int { return cascadeMinAMD64(kernels.UseStockhamAVX512, kernels.IntelCPU) }

// cascadeMinAMD64 is cascadeMinDefault's choice.
func cascadeMinAMD64(avx512, intel bool) int {
	if avx512 && intel {
		return 1 << 16
	}
	return 0
}

// splitTableDefault is splitTableAMD64, or intelSplitTable512 where the
// 512-bit split layout runs, for this machine.
func splitTableDefault() map[int][]int {
	return intelSplitTableAMD64(kernels.UseStockhamSplit, kernels.UseStockhamSplit512)
}

// intelSplitTableAMD64 is splitTableDefault's choice: intelSplitTable512 with
// the 512-bit split layout (kernels.UseStockhamSplit512), else
// splitTableAMD64.
func intelSplitTableAMD64(on, on512 bool) map[int][]int {
	if on512 {
		return intelSplitTable512()
	}
	return splitTableAMD64(on)
}

func init() { intelStripOrder = intelStripOrderAMD64(kernels.UseStockhamSplit512) }

// intelStripOrderAMD64 is intelStripOrder: the lengths of intelSplitTable512
// when the 512-bit split layout runs. On Cascade Lake (Round 28) the batched
// column passes of 1024×1024 ran 8·4·4·8 1.10× slower than 4·4·8·8 (five
// rotated rounds, 2-D 1024² 19.4 against 17.6 ms); the strips run
// interleaved whatever the layout, and the table was chosen for it.
func intelStripOrderAMD64(on512 bool) map[int]bool {
	if !on512 {
		return nil
	}
	m := map[int]bool{}
	for n := range intelSplitTable512() {
		m[n] = true
	}
	return m
}

// intelSplitTable512 gives the powers of two whose factorization changes
// with the 512-bit split layout (Round 28, Cascade Lake, one pinned core).
// Every ordering of radix-4, -8 and -16 passes from 256 to 16384 points was
// timed with the layout (three rounds), then the best few against the
// current factorization (seven rounds, fifteen for the close ones, the order
// rotated every round). Current factorization's time ÷ the table's: 1024
// (4·4·8·8 → 8·4·4·8) 1.036, every round above 1.017; 8192 (4·4·8·8·8 →
// 8·8·4·4·8) 1.045, every round above 1.030. 256, 2048 and 16384 had a
// candidate within 1% (8·4·8, 8·8·4·8, 8·8·4·8·8), so they keep theirs, as do
// 512, 4096 and 32768, where theirs was the best. Radix 16 lost at every size once the
// layout ran (intelRadix16Table512).
func intelSplitTable512() map[int][]int {
	return map[int][]int{
		1024: {8, 4, 4, 8},
		8192: {8, 8, 4, 4, 8},
	}
}

// intelRadix16Table512 gives the powers of two that take radix-16 passes on
// an Intel CPU with AVX-512: 128 = 16·8. Below 256 points the AVX-512
// kernels do not run (wide512), and the AVX2 radix-16 pass made 128 1.18×
// as fast as 4·4·8 on Cascade Lake (Round 28, fifteen rotated rounds, every
// one above 1.15; 8·16 tied 16·8). From 256 the 512-bit split layout runs
// and radix 16 lost to it at every size timed, 256 to 16384: the best
// factorization with a radix-16 pass took 1.08–1.55× the time of the split
// one (three rounds).
func intelRadix16Table512() map[int][]int { return map[int][]int{128: {16, 8}} }

// splitTableAMD64 gives the powers of two whose factorization changes when
// the split layout runs (kernels.UseStockhamSplit). Every ordering of radix-4
// and radix-8 passes closed by a radix-4, -8 or -16 pass was timed on Zen 3
// up to 16384 points, and the best few up to 2^20 (Round 23); below 65536
// the table takes the best of them, from 65536 splitPow2Factors, which was
// the best or within 3% of it there. 32, 64, 512 and 1024 keep skFactorize's
// factorization, which runs split as well as anything timed.
func splitTableAMD64(on bool) map[int][]int {
	if !on {
		return nil
	}
	t := map[int][]int{
		128:   {8, 16},
		256:   {4, 8, 8},
		2048:  {8, 8, 8, 4},
		4096:  {4, 8, 4, 8, 4},
		8192:  {4, 4, 4, 4, 8, 4},
		16384: {4, 8, 8, 4, 4, 4},
		32768: {8, 8, 8, 4, 4, 4},
	}
	for e := 16; e <= 30; e++ {
		t[1<<e] = splitPow2Factors(e)
	}
	return t
}
