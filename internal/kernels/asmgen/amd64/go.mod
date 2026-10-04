// The amd64 kernel generator is its own module so the go-asmgen version it
// generates with is pinned here, not resolved from whatever a cache holds:
// the committed .s files are reproducible only if the generator is.
module github.com/go-fft/fft/internal/kernels/asmgen/amd64

go 1.26.4

require github.com/go-asmgen/asmgen v0.12.0
