//go:build !amd64 && !arm64 && !s390x

package kernels

// On every architecture except amd64, arm64, and s390x there is no
// bit-identical generated SIMD kernel, so CMul uses CMulScalar directly and
// cmulSIMD aliases the scalar path. The SIMD-vs-scalar test then trivially holds
// (scalar == scalar) and still exercises this aliasing line, keeping the package
// building and the coverage gate green uniformly across all six targets.
//
// Which arches ship a SIMD kernel is decided by ONE rule: bit-for-bit equality
// with the scalar oracle, proven on that target. Three arches ship one —
// amd64 (SSE2 MULPD/ADDPD, matching its non-fused GOAMD64=v1 oracle), arm64
// (NEON VFMLA/VFMLS reproducing the oracle's fused FMSUBD/FMADDD), and s390x
// (vector-facility VFMDB/VFMSDB/VFMADB reproducing the oracle's fused
// FMSUB/FMADD on the project's only big-endian target). The three handled here —
// riscv64, loong64, ppc64le — keep the validated scalar path, each for a
// concrete, checked reason, NOT for lack of trying:
//
//   - riscv64: a genuine RVV kernel (VLSEG2E64V deinterleave + VFMACC.VV /
//     VFNMSAC.VV fused FMA, the exact analogue of the arm64 NEON kernel) was
//     written and assembles cleanly. But the V (vector) extension is not present
//     in the default qemu-riscv64 CPU that the CI's arch-qemu job runs, so the
//     kernel raises SIGILL there (VSETVLI is an illegal instruction); and
//     qemu's -cpu max RVV faults on the segment load. The project's rule is that
//     a kernel must be PROVEN bit-identical by its per-arch job — an RVV kernel
//     cannot even run there, and shipping it would turn CI red on real hardware-
//     less runners. So riscv64 stays scalar until the CI gains an RVV-capable
//     execution environment.
//
//   - loong64: the Go loong64 assembler exposes LSX vector FLOATING-POINT only
//     as unary ops (VFSQRTD, VFRINTD, VFRECIPD, …) — there is no vector float
//     add, multiply, or fused multiply-add to build a complex product from. A
//     bit-identical SIMD kernel is therefore not expressible; the scalar path
//     (which the compiler already fuses to FMSUBD/FMADDD) is the correct one.
//
//   - ppc64le: the Go ppc64 assembler exposes no vector DOUBLE arithmetic (the
//     VSX surface is loads/stores, logicals, permutes, and conversions —
//     XVMADDDP / XVMULDP / XVADDDP are not assemblable). As on loong64, a
//     bit-identical vector complex multiply cannot be built, so the validated
//     scalar path (compiler-fused FMSUB/FMADD) stands.
//
// All three are still exercised — endianness, word size, and the SIMD==scalar
// assertion (trivially scalar == scalar) — by their per-arch qemu jobs.

func cmulSIMD(a, b []complex128) { CMulScalar(a, b) }
