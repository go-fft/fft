//go:build !amd64

package kernels

// On every architecture except amd64 there is no bit-identical generated SIMD
// kernel, so CMul uses CMulScalar directly and cmulSIMD aliases the scalar
// path. The SIMD-vs-scalar test then trivially holds (scalar == scalar) and
// still exercises this aliasing line, keeping the package building and the
// coverage gate green uniformly across all six targets.
//
// Why not arm64/riscv64/loong64/ppc64le/s390x? The bottleneck is bit-for-bit
// equality with the scalar oracle, which computes ar*br - ai*bi as separately
// rounded multiplies followed by a rounded subtract. Go's arm64 assembler
// exposes only FUSED vector floating-point (VFMLA/VFMLS) — there is no
// non-fused vector VFMUL/VFADD/VFSUB — so a NEON complex multiply necessarily
// fuses a multiply-add and diverges from the scalar oracle by up to 1 ULP. The
// other targets' Go vector-float support is similarly FMA-centric. Rather than
// ship a kernel that produces different numbers (a correctness divergence, even
// if "more accurate"), these arches keep the validated scalar path. amd64's
// SSE2 kernel is bit-identical because MULPD+ADDPD are separately rounded, the
// same rounding the scalar oracle performs.

func cmulSIMD(a, b []complex128) { CMulScalar(a, b) }
