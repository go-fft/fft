// Separate benchmark module so the go-fft library itself stays
// dependency-free: gonum is pulled in here, never by the library go.mod.
// Run head-to-head benchmarks with:
//
//	cd bench && go test -run=^$ -bench=. -benchmem
module github.com/go-fft/fft/bench

go 1.26.4

require (
	github.com/go-fft/fft v0.0.0
	gonum.org/v1/gonum v0.16.0
)

replace github.com/go-fft/fft => ../
