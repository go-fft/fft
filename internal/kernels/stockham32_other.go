//go:build !amd64 && !arm64

package kernels

// StockhamTwiddles32 returns nils: no float32 pass kernel reads them.
func StockhamTwiddles32(r, ido, l1 int, root []complex128) (fwd, conj []complex64) {
	return nil, nil
}

// StockhamPass32 reports false: the fft package runs its float32 Go passes.
func StockhamPass32(r, ido, l1 int, cc, ch, tw []complex64, inverse bool) bool {
	return false
}
