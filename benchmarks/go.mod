// Self-contained benchmark module for the performance-parity report.
//
// Kept separate from the library go.mod so the go-fft library itself stays
// dependency-free (gonum is pulled in here, never by the library). Run the
// full parity sweep with ./run.sh, or the Go side alone with:
//
//	go test -run=^$ -bench=. -benchmem .
module github.com/go-fft/fft/benchmarks

go 1.26.4

require (
	github.com/go-fft/fft v0.0.0
	gonum.org/v1/gonum v0.16.0
)

replace github.com/go-fft/fft => ../
