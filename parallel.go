package fft

import (
	"runtime"
	"sync"
)

// This file centralizes the goroutine-parallel execution helpers. numpy.fft /
// pocketfft are single-threaded, so distributing the independent sub-transforms
// of a multi-dimensional transform (every row/column line along an axis is
// independent) across goroutines is where a pure-Go library can decisively beat
// single-threaded C — provided the work is large enough to amortize the
// scheduling cost. Below the threshold everything runs inline on the caller's
// goroutine so small transforms pay nothing.

// parWorkers is the goroutine count used by the parallel helpers, fixed at
// construction to GOMAXPROCS. A value <= 1 disables parallelism.
var parWorkers = runtime.GOMAXPROCS(0)

// parThreshold is the minimum total work (line count × per-line length, i.e.
// roughly the element count touched) below which parallelization is skipped.
// Spawning goroutines for tiny grids costs more than it saves; this floor was
// chosen so the cross-over sits where the parallel path measurably wins on the
// benchmark host.
const parThreshold = 1 << 14

// parChunks splits the half-open range [0,n) into up to parWorkers contiguous
// chunks and invokes body(lo, hi) for each, concurrently, blocking until all
// return. When parallelism is disabled, or n is small, or only one chunk would
// result, body is called once inline as body(0, n) — no goroutine is spawned.
// body must be safe to run concurrently across disjoint index ranges.
func parChunks(n int, body func(lo, hi int)) {
	w := parWorkers
	if w > n {
		w = n
	}
	if w <= 1 {
		body(0, n)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + w - 1) / w
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// parallelizeLines reports whether a set of lineCount independent transforms of
// length lineLen each is large enough to be worth running across goroutines.
func parallelizeLines(lineCount, lineLen int) bool {
	return parWorkers > 1 && lineCount > 1 && lineCount*lineLen >= parThreshold
}
