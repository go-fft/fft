package fft

import (
	"fmt"
	"strings"
)

// The bit-identity sweeps (every length to 2,100, both directions, every
// signal, in place and not) ran most of the suite's time on the hosted CI
// runners: Windows took 722 s of the package's 600-s budget (measured
// 2026-10-27, go test -json). Most of those lengths run the same pass
// kernels on the same code paths as a neighbour. routeSample keeps one length
// per route: per pass, the radix, the shape of its block (ido: each value to
// 8, then its residue mod 8, which decides the kernels' four- and eight-point
// groups, pairs and single-point tails) and of its block count (l1: to 4,
// then its parity), and per length the classes the plan depends on (a power
// of two from 256 and from 1,024 points, the frame kernels' 256 and 512).

// passKey describes one factorization of n as its passes run.
func passKey(n int, factors []int) string {
	var b strings.Builder
	l1 := 1
	for _, r := range factors {
		ido := n / (l1 * r)
		ic, lc := ido, l1
		if ido > 8 {
			ic = 8 + ido%8
		}
		if l1 > 4 {
			lc = 4 + l1%2
		}
		fmt.Fprintf(&b, "%d/%d/%d ", r, ic, lc)
		l1 *= r
	}
	return b.String()
}

// lengthClass is what a plan of n points depends on besides its passes.
func lengthClass(n int) string {
	p2 := n&(n-1) == 0
	return fmt.Sprintf("p2:%v,%v,%v small:%v,%v", p2, p2 && n >= 256, p2 && n >= 1024, n <= 256, n <= 512)
}

// routeSample keeps, in order, each size whose key no earlier size had.
func routeSample(sizes []int, key func(n int) string) []int {
	seen := map[string]bool{}
	var out []int
	for _, n := range sizes {
		k := key(n)
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, n)
	}
	return out
}

// smoothRoute keys a smooth length by the given factorizations of it, and
// every other length by itself (each is kept).
func smoothRoute(factorings func(n int) [][]int) func(n int) string {
	return func(n int) string {
		if !factorsAreSmall(n) {
			return fmt.Sprintf("n=%d", n)
		}
		k := lengthClass(n)
		for _, f := range factorings(n) {
			k += "|" + passKey(n, f)
		}
		return k
	}
}

// upTo is 2, 3, …, hi.
func upTo(hi int) []int {
	s := make([]int, 0, hi)
	for n := 2; n <= hi; n++ {
		s = append(s, n)
	}
	return s
}
