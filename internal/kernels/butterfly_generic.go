//go:build !amd64 && !arm64 && !s390x && !riscv64

package kernels

// loong64 and ppc64le expose no vector DOUBLE arithmetic in the Go assembler
// (see cmul_generic.go for the detailed reason), so like arm64/s390x/riscv64
// they run the butterfly STAGES through the gc-autovectorized plain Go loop, not
// a hand-written kernel. The bodies are identical to Radix2StageScalar/
// Radix4StageScalar and are deliberately inlinable so the autovectorizer
// optimizes them in the caller; the SIMD-vs-scalar test trivially holds and the
// transform result is unchanged. amd64 alone ships a routed SSE2 kernel.

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
