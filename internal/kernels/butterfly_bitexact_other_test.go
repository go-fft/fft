//go:build !amd64

package kernels

// Off amd64 the butterfly stage runs an inlinable Go loop the gc compiler
// autovectorizes IN THE CALLER. On the FMA arches (arm64/s390x/riscv64) the
// compiler may fuse a complex product with a following radix add into one FMA,
// which rounds up to 1 ULP differently than the noinline scalar oracle (whose
// pinned cmul1 keeps every product a separate rounding). That is a legitimate,
// numerically-correct rounding difference, not a kernel bug — the transform is
// still validated against FFTW/numpy within tolerance. So the SIMD-vs-scalar
// test requires agreement with the oracle within a tight ULP tolerance here, not
// bit-for-bit equality (which is asserted only on amd64, where the kernel is
// hand-written assembly with a no-FMA bit-identity contract).
const butterflyBitExact = false
