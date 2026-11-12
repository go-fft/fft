//go:build !amd64 && !arm64 && !s390x && !riscv64

package kernels

// On loong64 and ppc64le (and any other architecture without a kernel here)
// CMul uses CMulScalar directly and cmulSIMD aliases the scalar path. The
// SIMD-vs-scalar test then trivially holds (scalar == scalar) and still exercises
// this aliasing line, keeping the package building and the coverage gate green
// uniformly across all six targets.
//
// amd64, arm64, s390x and riscv64 ship a kernel, each held bit for bit to the
// scalar oracle by its CI job: amd64 (SSE2 MULPD/ADDPD, matching its non-fused
// GOAMD64=v1 oracle), arm64 (NEON VFMLA/VFMLS reproducing the oracle's fused
// FMSUBD/FMADDD), s390x (vector-facility VFMDB/VFMSDB/VFMADB, on the project's
// only big-endian target), and riscv64 (RVV 1.0 VLSEG2E64V + VFMUL.VV/VFNMSAC.VV/
// VFMACC.VV, with run-time V-extension detection; hardware-validated on
// cfarm95). The two handled here keep the scalar path for the reasons below.
//
//   - loong64: a vector kernel is expressible. The Go loong64 assembler has had
//     vector double add, subtract and multiply (VADDD/VSUBD/VMULD =
//     vfadd.d/vfsub.d/vfmul.d) since Go 1.26 (assembled with Go 1.27.4), and it
//     has no vector fused multiply-add (vfmadd.d is not assemblable, checked with
//     Go 1.27.1, not re-checked since). A kernel from separately rounded
//     products differs from the compiler-fused scalar path by an ULP, which the
//     accuracy contract (docs/accuracy.md, go-fft/fft#121) allows inside the
//     bound, but none is built: nothing measured on loong64 shows a gain, and
//     the scalar loop measured as fast as the kernels on the other
//     architectures (README, BENCHMARKS.md). It is step 3 of #121, tracked in
//     go-fft/fft#137.
//
//   - ppc64le: no vector complex multiply can be built, bit-identical or not:
//     the Go ppc64 assembler exposes no vector DOUBLE arithmetic. The VSX surface
//     is loads/stores, logicals, permutes and conversions; XVMADDDP, XVMULDP and
//     XVADDDP are not assemblable (XVMULDP re-checked with Go 1.27.4:
//     "unrecognized instruction"). The scalar path (compiler-fused FMSUB/FMADD)
//     stands until the assembler gains them.
//
// Both are still exercised — endianness, word size, and the SIMD==scalar
// assertion (trivially scalar == scalar) — by their per-arch qemu jobs.

func cmulSIMD(a, b []complex128) { CMulScalar(a, b) }
