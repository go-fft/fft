//go:build !amd64 && !arm64

package kernels

// On every architecture except amd64 and arm64 there is no bit-identical
// generated SIMD kernel yet, so CMul uses CMulScalar directly and cmulSIMD
// aliases the scalar path. The SIMD-vs-scalar test then trivially holds (scalar
// == scalar) and still exercises this aliasing line, keeping the package
// building and the coverage gate green uniformly across all six targets.
//
// Why not riscv64/loong64/ppc64le/s390x? The contract is bit-for-bit equality
// with the scalar oracle, which on each target the gc compiler may compile to a
// target-specific (often FMA-fused) form. amd64 (SSE2 MULPD/ADDPD, non-fused
// oracle) and arm64 (NEON VFMLA/VFMLS reproducing the oracle's fused FMSUBD/
// FMADDD) each ship a validated, bit-identical kernel. The remaining four
// qemu-only targets keep the validated scalar path until a kernel is built and
// PROVEN bit-identical by their per-arch execution job — shipping a kernel that
// produces even 1-ULP-different numbers would be a correctness divergence, which
// the project does not accept. These arches are still exercised (endianness,
// word size, scalar==scalar) by the qemu jobs, s390x covering big-endian.

func cmulSIMD(a, b []complex128) { CMulScalar(a, b) }
