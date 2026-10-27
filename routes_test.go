package fft

import (
	"fmt"
	"slices"
	"testing"
)

// The bit-identity sweeps (every length to 2,100, every signal, both
// directions, in place and not) took most of the suite's time on the hosted
// CI runners: on Windows the package took 722 s against go test's 600-s
// default (go test -race -cover -json, 2026-10-27), 239 s of it in
// TestStockham32PassMatchesScalar alone. Most of those lengths run passes
// that smaller lengths already ran. routeSample keeps a length only when it
// brings something no smaller kept length brought: a pass route (the radix,
// the shape of its block, its place in the plan) or a length class.

// passKeys describes each pass of one factorization of n as the kernels
// branch on it: the radix; ido as 1 (the final pass), 2, 3, 4, then its
// residue mod 4 (the four-point groups' pair and single-point tails) or a
// multiple of 8 (whole eight-point groups, the 512-bit split layout); l1 as
// one block or many, and in the final pass its residue mod 4 (the block
// pairs and quads and their tails); the pass's place (first, between, last)
// and the pass count, which decide the split layout's modes and the
// ping-pong buffers.
func passKeys(n int, factors []int) []string {
	var keys []string
	l1 := 1
	for k, r := range factors {
		ido := n / (l1 * r)
		ic := ido
		if ido > 4 {
			ic = 4 + ido%4
			if ido%8 == 0 {
				ic = 12
			}
		}
		lc := min(l1, 2)
		if ido == 1 {
			lc = 4 + l1%4
		}
		pos := "mid"
		switch {
		case k == 0:
			pos = "first"
		case k == len(factors)-1:
			pos = "last"
		}
		keys = append(keys, fmt.Sprintf("%d/%d/%d/%s/%d", r, ic, lc, pos, len(factors)))
		l1 *= r
	}
	return keys
}

// lengthClass is what a plan of n points depends on besides its passes:
// powers of two (from 256 points the AVX-512 kernels, from 1,024 the
// scratch gap), and the frame kernels' 256 and 512 points.
func lengthClass(n int) string {
	p2 := n&(n-1) == 0
	return fmt.Sprintf("class p2:%v,%v,%v small:%v,%v", p2, p2 && n >= 256, p2 && n >= 1024, n <= 256, n <= 512)
}

// routeSample keeps, in order, each size whose keys include one that no
// earlier kept size had.
func routeSample(sizes []int, keys func(n int) []string) []int {
	seen := map[string]bool{}
	var out []int
	for _, n := range sizes {
		fresh := false
		for _, k := range keys(n) {
			if !seen[k] {
				seen[k], fresh = true, true
			}
		}
		if fresh {
			out = append(out, n)
		}
	}
	return out
}

// smoothRoutes keys a smooth length by its class and the passes of the
// given factorizations of it; the other lengths, which the Stockham sweeps
// skip, have one key between them.
func smoothRoutes(factorings func(n int) [][]int) func(n int) []string {
	return func(n int) []string {
		if !factorsAreSmall(n) {
			return []string{"not smooth"}
		}
		keys := []string{lengthClass(n)}
		for _, f := range factorings(n) {
			keys = append(keys, passKeys(n, f)...)
		}
		return keys
	}
}

// oneFactoring is skFactorize's factorization as smoothRoutes takes it.
func oneFactoring(n int) [][]int { return [][]int{skFactorize(n)} }

// upTo is 2, 3, …, hi.
func upTo(hi int) []int {
	s := make([]int, 0, hi)
	for n := 2; n <= hi; n++ {
		s = append(s, n)
	}
	return s
}

// TestRouteSample: routeSample keeps a size exactly when it brings a new
// key, and the sweeps' samples (logged) keep every pass route of the full
// range.
func TestRouteSample(t *testing.T) {
	got := routeSample([]int{2, 3, 4, 5, 6, 7}, func(n int) []string { return []string{fmt.Sprint("a", n%2), fmt.Sprint("b", n%3)} })
	if !slices.Equal(got, []int{2, 3, 4}) {
		t.Fatalf("routeSample = %v, want [2 3 4]", got)
	}
	keys := smoothRoutes(oneFactoring)
	s := routeSample(upTo(2100), keys)
	all, kept := map[string]bool{}, map[string]bool{}
	for _, n := range upTo(2100) {
		for _, k := range keys(n) {
			all[k] = true
		}
	}
	for _, n := range s {
		for _, k := range keys(n) {
			kept[k] = true
		}
	}
	if len(kept) != len(all) {
		t.Fatalf("the sample keeps %d of %d routes", len(kept), len(all))
	}
	t.Logf("%d of 2099 lengths to 2,100 keep all %d routes", len(s), len(all))
}
