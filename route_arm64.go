package fft

// pow2StockhamDefault routes powers of two to the Stockham engine on arm64,
// where it measured faster than the iterative pow2 kernel at every size (M4
// Max: 256 1.07×, 1024 1.30×, 4096 1.10×, 65536 1.34×). See route_other.go
// for why every other architecture keeps the pow2 kernel.
const pow2StockhamDefault = true
