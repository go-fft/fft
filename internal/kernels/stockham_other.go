//go:build !amd64

package kernels

// StockhamTwiddles returns nils: no pass kernel reads them.
func StockhamTwiddles(r, ido, l1 int, root []complex128) (fwd, conj []complex128) { return nil, nil }

// StockhamPass reports false: the fft package runs its scalar passes.
func StockhamPass(r, ido, l1 int, cc, ch, tw []complex128, inverse bool) bool { return false }

// Untangle returns 0: the fft package untangles every bin in Go.
func Untangle(dst, z, tw []complex128, m int) int { return 0 }

// Retangle returns 0: the fft package rebuilds every bin in Go.
func Retangle(z, x, tw []complex128, m int, h float64) int { return 0 }
