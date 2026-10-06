package fft

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

// compFactorize is skFactorize past radix16Table.
func compFactorize(n int) []int { return skFactorizeOrder(n, compOddFirst) }
