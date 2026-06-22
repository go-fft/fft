package kernels

// Vectorized decimation-in-time butterfly STAGES — the FFT hot loop.
//
// The iterative power-of-two engine (the fft package's iterative.go) runs log-N
// sequential passes over the data; each pass (stage) is a sweep of radix-2 or
// radix-4 butterflies over the whole array. Those passes are the single hottest
// code in the library, and the documented gap versus FFTW on the power-of-two
// and smooth-composite mid-range is exactly that FFTW issues two complex
// multiplies per SIMD instruction in its hand-written codelets while a scalar Go
// loop does one at a time (see BENCHMARKS.md, action item 1).
//
// These kernels operate at STAGE granularity — the whole pass, both the outer
// loop over groups and the inner loop over butterfly positions, lives inside one
// kernel call. That matters: a per-GROUP kernel call pays one Go call per group,
// and the early stages have a tiny span and therefore very many groups
// (n/(radix·span) calls), which swamps any SIMD win. One call per stage has
// zero per-group overhead, so the kernel competes with the gc-autovectorized
// inline loop on equal footing and wins only on real SIMD throughput.
//
// Data layout. Within a group the operand blocks are contiguous (radix-2: the
// even block a[base:base+span] and the odd block a[base+span:base+2·span];
// radix-4: the four blocks a[i0:i0+span], a[i1:..], a[i2:..], a[i3:..] with
// i1=i0+span etc.), so each is a unit-stride run the SIMD load can stream. The
// twiddles are stored as separate contiguous planes (radix-2: w[0:span];
// radix-4: w1|w2|w3, each span long) and are re-read from the plane start for
// every group.
//
// Bit-identity contract. As with cmul, each per-arch SIMD kernel must be
// bit-for-bit identical to the scalar oracle in this file, proven by that arch's
// CI execution job. The complex multiplies are the only place fused-multiply-add
// can diverge across arches: amd64's GOAMD64=v1 oracle does NOT fuse
// (separately-rounded MULPD/ADDPD match it), while arm64, s390x and riscv64 fuse,
// so their kernels must reproduce that exact fusion. The radix add/subs and the
// ∓i rotation are exact on every arch. Where no bit-identical kernel is
// expressible (loong64/ppc64le, no vector double arithmetic in the Go assembler)
// the scalar oracle stands.

// Radix2StageScalar runs one radix-2 decimation-in-time pass in place over a
// (length n). span is the sub-transform length entering the stage; groups of
// 2·span are combined. tw is the span-long twiddle plane (tw[k]=W_{2·span}^k).
// For each group base and each k in [0,span):
//
//	t            = a[base+span+k] * tw[k]
//	a[base+k]    = a[base+k] + t
//	a[base+span+k] = a[base+k]_old - t
//
// It is the portable correctness oracle and the fallback when no SIMD kernel is
// selected. //go:noinline pins one compiled body so every caller observes the
// same floating-point form (the same reason CMulScalar is noinline).
//
//go:noinline
func Radix2StageScalar(a []complex128, n, span int, tw []complex128) {
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

// Radix4StageScalar runs one radix-4 decimation-in-time pass in place over a
// (length n). span is the sub-transform length entering the stage; groups of
// 4·span are combined. w1,w2,w3 are the three span-long twiddle planes
// (w1[k]=W^k, w2[k]=W^{2k}, w3[k]=W^{3k}). inverse selects the rotation (×−i
// forward, ×+i inverse). For each group base and each k in [0,span):
//
//	m1 = a[i1+k]*w1[k], m2 = a[i2+k]*w2[k], m3 = a[i3+k]*w3[k]   (i1=base+span …)
//	t0 = a[i0+k]+m2, t1 = a[i0+k]-m2, t2 = m1+m3, t3 = rot∓i(m1-m3)
//	a[i0+k]=t0+t2, a[i1+k]=t1+t3, a[i2+k]=t0-t2, a[i3+k]=t1-t3
//
//go:noinline
func Radix4StageScalar(a []complex128, n, span int, w1, w2, w3 []complex128, inverse bool) {
	step := 4 * span
	for base := 0; base < n; base += step {
		i0 := base
		i1 := i0 + span
		i2 := i1 + span
		i3 := i2 + span
		for k := 0; k < span; k++ {
			m1 := cmul1(a[i1+k], w1[k])
			m2 := cmul1(a[i2+k], w2[k])
			m3 := cmul1(a[i3+k], w3[k])
			a0 := a[i0+k]
			t0 := a0 + m2
			t1 := a0 - m2
			t2 := m1 + m3
			d := m1 - m3
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

// cmul1 is the single-element complex product, written exactly as CMulScalar's
// body so the radix oracles fuse identically to the pointwise oracle on the
// fused arches.
//
// //go:noinline pins the product as its OWN rounding boundary. Without it the
// gc compiler inlines cmul1 into Radix4StageScalar and may then fuse a product
// term with the following radix add (e.g. m1_im + m3_im) into a single FMA on
// the fused arches, rounding differently than the SIMD kernel — which computes
// each product, then each add, as separate roundings. Pinning the product keeps
// the scalar oracle's rounding structure identical to the kernel's, preserving
// bit-for-bit identity (the same FMA-contraction hazard CMulScalar documents).
//
//go:noinline
func cmul1(a, b complex128) complex128 {
	ar, ai := real(a), imag(a)
	br, bi := real(b), imag(b)
	return complex(ar*br-ai*bi, ar*bi+ai*br)
}
