//go:build !arm64 && !amd64

package fft

// compOddFirstDefault is the order the float32 plans take (oddRadicesFirst):
// the complex128 order was only re-measured on amd64 and arm64 (Round 24).
func compOddFirstDefault() bool { return oddRadicesFirstDefault() }

// compFactorize is skFactorize past radix16Table.
func compFactorize(n int) []int { return skFactorizeOrder(n, compOddFirst) }
