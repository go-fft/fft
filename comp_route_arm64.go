package fft

import (
	"math/bits"
	"slices"
)

// compOddFirstDefault is true on arm64: a complex128 transform orders its
// radices odd first (3s and 5s, then the powers of two with radix 4 before 8,
// then 7, 11, 13), as amd64 has since Round 17. With that order every pass
// before the last keeps a factor 2 in its ido, so a length 2^e·3^a·5^b keeps
// its data block-split between passes (kernels.StockhamSplitModes, Round 21's
// layout), and its final pass is a radix-2, 4 or 8 NEON pass over an odd
// number of blocks, which ends with one block alone. On Neoverse-N1 (Round 24,
// one pinned core, R24Decomp medians of five rounds) the whole transform went
// from the pocketfft order's 7,359 / 8,797 / 11,754 / 15,070 / 16,937 /
// 61,128 ns to 6,048 / 6,934 / 8,615 / 12,054 / 13,771 / 49,582 ns at 1000 /
// 1080 / 1296 / 1920 / 2000 / 6000 points. float32 plans keep their own order
// (oddRadicesFirst, see plan32.go): their final passes need a multiple of
// four blocks.
func compOddFirstDefault() bool { return true }

// compFactorize is skFactorize past radix16Table: for a complex128 power of
// two up to smallArmR8Max, radix8Maximal's passes, radix 4 first
// (smallArmPow2); for every other length skFactorizeOrder's.
func compFactorize(n int) []int {
	if f := smallArmPow2(n); f != nil {
		return f
	}
	return skFactorizeOrder(n, compOddFirst)
}

// smallArmR8Max is the largest power of two smallArmPow2 factors.
const smallArmR8Max = 4096

// smallArmPow2 factors a power of two 2^e from 64 to smallArmR8Max points as
// radix-8 passes with one or two radix-4 passes first (radix8Maximal,
// reversed: e mod 3 = 0 all radix 8, 2 one radix 4, 1 two), or returns nil.
//
// The arm64 rule for powers of two (r8MaxPow2Default, route_arm64.go) was
// timed in 2026-09 on the interleaved Go and then NEON passes: radix 4 with
// at most one radix-8 pass. Since Round 21 a power of two keeps its data
// split between NEON passes, and on Neoverse-N1 (Round 30, every ordering of
// radix-2, 4 and 8 passes timed, one pinned core, seven rounds) the radix-8
// passes now win from 64 to 4096 points: rule time ÷ this factorization's
// 1.17 (64), 1.10 (256), 1.07 (512, 1024), 1.09 (2048), 1.14 (4096), each
// within 0.5% of the best ordering timed but at 1024 (1.073 against 1.075).
// 128, 32 and below already factored this way; 8192 did not move (1.006).
// float32 plans keep the old rule: their passes were not timed.
func smallArmPow2(n int) []int {
	if n < 64 || n > smallArmR8Max || n&(n-1) != 0 {
		return nil
	}
	f := radix8Maximal(bits.TrailingZeros(uint(n)))
	slices.Reverse(f)
	return f
}
