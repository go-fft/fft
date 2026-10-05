//go:build arm64 || s390x || riscv64

package kernels

// arm64, s390x and riscv64 run the butterfly STAGES through a plain Go loop —
// NOT a hand-written SIMD kernel. gc compiles it to scalar code (it does not
// vectorize; on these arches it fuses multiply-adds). A MEASURED decision, not
// a gap in effort:
//
//   - When this was measured (Go 1.26), the Go arm64 and s390x assemblers exposed
//     vector floating-point only as the fused multiply-add family (no vector
//     VFADD/VFSUB; VADD/VSUB are integer), so a vector butterfly had to emulate
//     every add/sub as a copy plus an FMA-by-one and pay a VLD2/VST2 (or VL/VST +
//     VMRH/VMRL) deinterleave on each pass. A full stage-level NEON kernel was
//     built and benchmarked against this loop: it TIED the scalar loop (≈±2%).
//     (Same as the SIMD complex-multiply round; see cmul.go and BENCHMARKS.md.
//     This comment used to credit a gc "autovectorizer"; gc has none, the
//     scalar loop is simply as fast as that kernel.)
//   - These stages only serve the iterative pow2 kernel, which arm64 no longer
//     uses for transforms (route_arm64.go sends every power of two to the
//     Stockham engine). The Stockham passes DO run as NEON kernels on arm64
//     since Go 1.27 assembles VFADD/VFSUB/VFMUL/VFNEG (stockham_arm64.go,
//     1.08–1.83× per pass on Neoverse-N1, Round 18 of BENCHMARKS.md); this
//     stage-level kernel was not built again.
//   - riscv64's RVV is run-time-optional and its strip-mined kernel was not
//     measured to beat this loop on the available hardware either.
//
// These functions are deliberately INLINABLE (no //go:noinline): they are
// compiled in their caller exactly as the loop was when it lived inline in the
// fft package, and match the inline-loop speed (routing through the noinline
// scalar ORACLE instead, Radix*StageScalar, measured ~2× slower; the cause was
// not established — both bodies are scalar code). The
// loop bodies are identical to Radix2StageScalar/Radix4StageScalar, so the
// SIMD-vs-scalar test (which compares against the oracle) trivially holds and the
// transform result is unchanged. amd64 is the one arch whose SSE2 stage kernel
// (real packed ADDPD/SUBPD, no emulation tax, against a loop gc compiles to
// scalar SSE2 under GOAMD64=v1) was measured to win, and it alone ships a routed
// asm kernel.

func radix2StageSIMD(a []complex128, n, span int, tw []complex128) {
	step := 2 * span
	for base := 0; base < n; base += step {
		for k := 0; k < span; k++ {
			i := base + k
			j := i + span
			hr, hi := real(a[j]), imag(a[j])
			wr, wi := real(tw[k]), imag(tw[k])
			t := complex(hr*wr-hi*wi, hr*wi+hi*wr)
			u := a[i]
			a[i] = u + t
			a[j] = u - t
		}
	}
}

func radix4StageSIMD(a []complex128, n, span int, w1, w2, w3 []complex128, inverse bool) {
	step := 4 * span
	for base := 0; base < n; base += step {
		i0 := base
		i1 := i0 + span
		i2 := i1 + span
		i3 := i2 + span
		for k := 0; k < span; k++ {
			b1 := a[i1+k] * w1[k]
			b2 := a[i2+k] * w2[k]
			b3 := a[i3+k] * w3[k]
			a0 := a[i0+k]
			t0 := a0 + b2
			t1 := a0 - b2
			t2 := b1 + b3
			d := b1 - b3
			var t3 complex128
			if inverse {
				t3 = complex(-imag(d), real(d)) // ×(+i)
			} else {
				t3 = complex(imag(d), -real(d)) // ×(−i)
			}
			a[i0+k] = t0 + t2
			a[i1+k] = t1 + t3
			a[i2+k] = t0 - t2
			a[i3+k] = t1 - t3
		}
	}
}
