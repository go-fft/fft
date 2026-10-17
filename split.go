package fft

// The split layout's factorizations (Round 23). Where the pass kernels keep a
// power of two's data block-split between passes (kernels.StockhamSplitModes,
// on amd64 with AVX2 and without AVX-512), a few factorizations run faster
// than the ones skFactorize picks for the interleaved passes: splitTable
// holds them. It is per-architecture (route_amd64.go; nil elsewhere).
var splitTable = splitTableDefault()

// splitPow2Factors is the split layout's factorization of 2^e (e >= 4): a
// final radix-4 pass, radix-8 passes for as many of the other bits as leave
// an even remainder, and radix-4 passes for that remainder, first. Every
// pass but the last then runs split, and the final pass, which reads
// interleaved data, is the cheapest one per point. Timed on Zen 3 against
// every ordering of radix-4 and radix-8 passes closed by a radix-4, -8 or
// -16 pass (Round 23), it was the best or within 3% of it from 65536 points
// up; below, where a pass's data stays in L2, other orders won by up to 7%.
func splitPow2Factors(e int) []int {
	rest := e - 2
	n8 := rest / 3
	if (rest-3*n8)%2 != 0 {
		n8--
	}
	var f []int
	for range (rest - 3*n8) / 2 {
		f = append(f, 4)
	}
	for range n8 {
		f = append(f, 8)
	}
	return append(f, 4)
}

// intelStripOrder lists the lengths whose N-D strip axes (every axis but the
// last) take skFactorizeOrder's factorization, as cachedPlanNo16 builds it,
// although splitTable gives their 1-D plans another one: the batched passes
// of a strip run interleaved, and those factorizations were chosen for the
// split layout's 1-D passes. Set per architecture (route_amd64.go: Round 28's
// table); nil elsewhere.
var intelStripOrder map[int]bool

// clRowOrder gives, per length, the factorization the contiguous last axis of
// a large N-D plan (at least clRowMin elements) takes instead of its 1-D
// plan's: the rows of such an array come from beyond the cache, where a first
// pass that the hardware prefetchers follow badly costs more than the 1-D
// table's order gains in cache, or, on Haswell, where the 1-D plan's odd
// pass count costs an in-place transform a copy. Set per architecture
// (route_amd64.go: Round 31's table on Cascade Lake, Round 37's on
// Haswell); nil elsewhere.
var clRowOrder map[int][]int

// clRowMin is the smallest N-D plan, in elements, whose last axis takes
// clRowOrder, set with it per architecture: 2^19 (8 MB of complex128) on
// Cascade Lake, where (Round 31, m×1024 arrays,
// the row step alone, eleven rotated rounds) 4·8·4·8 rows ran 0.94–0.99× as
// fast as the 1-D table's 8·4·4·8 up to 128 rows (2 MB), tied at 256 rows
// (0.999) and gained 1.18 from 512 rows; 2^16 (64 rows of 1024) on
// Haswell (Round 37: route_amd64.go).
var clRowMin int

// clRowPlan returns the plan the last axis of an N-D plan of size elements
// takes for lines of length n: clRowOrder's when the plan is large enough
// and the length is in the table, else nil.
func clRowPlan(n, size int) *Plan { return clRowPlanF(n, size, false) }

// clRowPlanF is clRowPlan for an FMA plan when fma is set, from a cache of
// its own.
func clRowPlanF(n, size int, fma bool) *Plan {
	f := clRowOrder[n]
	if f == nil || size < clRowMin {
		return nil
	}
	cache := clRowCache
	if fma {
		cache = clRowCacheFMA
	}
	planMu.Lock()
	defer planMu.Unlock()
	p, ok := cache[n]
	if !ok {
		p = &Plan{n: n, sk: newSKPlanFactorsOpt(n, f, fma)}
		cache[n] = p
	}
	return p
}

var clRowCache, clRowCacheFMA = map[int]*Plan{}, map[int]*Plan{}

// hswSplitFloor, when positive, keeps the split layout to the powers of two
// of at least that many points: every other plan runs its passes
// interleaved. Set per architecture (route_amd64.go: 512 on an Intel CPU
// with the 256-bit layout, Round 29); 0 elsewhere.
var hswSplitFloor int

// hswSplitKept reports whether a plan of n points keeps the split modes
// kernels.StockhamSplitModes gave it, under hswSplitFloor.
func hswSplitKept(n int) bool {
	return hswSplitFloor <= 0 || n >= hswSplitFloor && n&(n-1) == 0
}

// cl3WideStrips makes the strip axes of a large N-D array take wider strips
// (cl3StripWidth). Set per architecture (route_amd64.go: Intel with the
// 512-bit split layout, Round 39); false elsewhere.
var cl3WideStrips bool

// cl3StripWidth is how many neighbouring lines one strip holds on an axis of
// length n whose lines start st elements apart. With cl3WideStrips, an axis
// spanning at least 2^18 elements (4 MB of complex128) whose lines lie at
// least 256 elements (4 KB) apart takes strips of 128 lines up to n = 256
// and of 64 above; every other axis takes stripWidth(n).
//
// On Cascade Lake (Round 39, the strip step alone, one process, seven
// rotated rounds, TestCl3StripVars), main's width ÷ the wide one:
//   - 512×512 1.93–2.36 (64; 128: 1.75), 512×1024 1.82–2.00,
//     512×2048 2.01, 640² 2.16, 1024² 1.54 (64; 128: 1.68),
//     1024×512 1.43, 2048² 1.40, 4096×256 1.45, 256×1024 1.14;
//   - n ≤ 256, 128 lines: 64×4096 2.12, 128×4096 2.07, 32×8192 1.40,
//     16×16384 1.36, the first two axes of 128³ 2.25 and 2.76, the first
//     of 64³ 2.97 and of 256×256×64 2.06;
//   - below the span or the distance the wide strips lost or tied:
//     384×512 and 256×768 (3 MB) 0.96 and 0.93, 512×384 0.90, 128² 0.87,
//     1024×128 and 2048×64 (lines 2 and 1 KB apart, from 16 lines)
//     0.78–0.91, 4096×64 1.00.
//
// The strip's first pass reads, and its last writes, a few hundred bytes
// of every line; once the array is beyond the cache, 1 KB or 2 KB of each
// line per visit is what the prefetchers follow (a probe that only reads
// the 512×512 columns' 256-byte pieces took 0.34 ms where the first pass
// took 1.31). This VM has no PMU to show it.
func cl3StripWidth(n, st int) int {
	if cl3WideStrips && n*st >= 1<<18 && st >= 256 {
		if n <= 256 {
			return 128
		}
		return 64
	}
	return stripWidth(n)
}
