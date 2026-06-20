// Separate module: the go-asmgen generators are build-time tools only. Keeping
// them out of the main github.com/go-fft/fft module means the FFT library has
// zero third-party dependencies, while the generators can still import
// go-asmgen to (re)produce the committed .s files via `go generate`.
module github.com/go-fft/fft/internal/kernels/asmgen

go 1.26.4

require github.com/go-asmgen/asmgen v0.5.0
