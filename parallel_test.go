package fft

import (
	"sync"
	"testing"
)

// withWorkers runs fn with parWorkers temporarily set to w (and parThreshold
// to 16384), restoring both after.
// It lets the parallel and serial branches both be exercised deterministically
// regardless of the host core count, so the coverage gate does not depend on
// GOMAXPROCS.
func withWorkers(w int, fn func()) {
	saved, th := parWorkers, parThreshold
	// The tests' grids were sized for the 16384-element threshold; arm64's is
	// higher (route_arm64.go), and would leave some of them on one goroutine.
	parWorkers, parThreshold = w, 1<<14
	defer func() { parWorkers, parThreshold = saved, th }()
	fn()
}

// TestParChunksCoversRange checks that parChunks visits every index exactly once,
// for both the inline (w<=1) and multi-goroutine (w>1) paths, including a worker
// count exceeding n (clamped) and a chunk boundary that does not divide evenly.
func TestParChunksCoversRange(t *testing.T) {
	for _, tc := range []struct{ n, w int }{
		{0, 1}, {1, 1}, {7, 1}, {7, 3}, {10, 4}, {5, 8}, {16, 2}, {17, 4},
	} {
		hits := make([]int32, tc.n)
		withWorkers(tc.w, func() {
			parChunks(tc.n, parMinChunk, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					hits[i]++
				}
			})
		})
		for i, h := range hits {
			if h != 1 {
				t.Fatalf("n=%d w=%d index %d hit %d times", tc.n, tc.w, i, h)
			}
		}
	}
}

// TestParChunksCapsWorkersByWork checks that parChunks never hands a goroutine
// less than parMinChunk elements of work, however many workers are available:
// on a 128-thread host a 128×128 transform used to run as 128 goroutines of
// one 128-point line each, 8.5× slower than FFTW (see BENCHMARKS.md).
func TestParChunksCapsWorkersByWork(t *testing.T) {
	for _, tc := range []struct{ n, itemLen, w, wantChunks int }{
		{128, parMinChunk / 64, 128, 2}, // two chunks of work: two chunks, not 128
		{100, 1, 8, 1},                  // far below one chunk: inline
		{64, parMinChunk / 8, 32, 8},    // work for 8 chunks, 32 workers
		{64, parMinChunk, 4, 4},         // plenty of work: the worker count rules
		{3, 10 * parMinChunk, 16, 3},    // few heavy items: one chunk each
	} {
		var mu sync.Mutex
		chunks := 0
		withWorkers(tc.w, func() {
			parChunks(tc.n, tc.itemLen, func(lo, hi int) {
				mu.Lock()
				chunks++
				mu.Unlock()
			})
		})
		if chunks != tc.wantChunks {
			t.Errorf("n=%d itemLen=%d workers=%d: %d chunks, want %d", tc.n, tc.itemLen, tc.w, chunks, tc.wantChunks)
		}
	}
}

// TestParallelizeLinesThreshold exercises both sides of the work-size gate.
func TestParallelizeLinesThreshold(t *testing.T) {
	withWorkers(4, func() {
		if parallelizeLines(1, 1<<20) {
			t.Error("single line must never parallelize")
		}
		if parallelizeLines(2, 1) {
			t.Error("tiny work must not parallelize")
		}
		if !parallelizeLines(parThreshold, 1) {
			t.Error("work at threshold should parallelize")
		}
	})
	withWorkers(1, func() {
		if parallelizeLines(1000, 1000) {
			t.Error("single worker must never parallelize")
		}
	})
}

// TestNDParallelMatchesSerial runs FFTN/FFT2/RFFT2/IRFFT2 under both the forced
// parallel and forced serial execution paths and requires identical results, so
// the goroutine fan-out is proven correct (and both branches are covered)
// independent of the host's core count. The grids are sized above parThreshold so
// the parallel branch is genuinely taken when workers > 1.
func TestNDParallelMatchesSerial(t *testing.T) {
	shapes := [][]int{{256, 256}, {128, 200}, {300, 1, 60}}
	for _, shape := range shapes {
		x := complexGrid(shape)

		var par, ser []complex128
		withWorkers(4, func() { par = FFTN(x, shape) })
		withWorkers(1, func() { ser = FFTN(x, shape) })
		closeVec(t, par, ser)

		// Round trip under the parallel path.
		withWorkers(4, func() { closeVec(t, IFFTN(FFTN(x, shape), shape), x) })
	}

	// 2-D real path: RFFT2/IRFFT2 with a grid above threshold.
	rows, cols := 256, 256
	rx := make([]float64, rows*cols)
	for i := range rx {
		rx[i] = float64((i*7+1)%13) * 0.1
	}
	var parC, serC []complex128
	withWorkers(4, func() { parC = RFFT2(rx, [2]int{rows, cols}) })
	withWorkers(1, func() { serC = RFFT2(rx, [2]int{rows, cols}) })
	closeVec(t, parC, serC)

	var parR, serR []float64
	withWorkers(4, func() { parR = IRFFT2(parC, [2]int{rows, cols}) })
	withWorkers(1, func() { serR = IRFFT2(serC, [2]int{rows, cols}) })
	for i := range parR {
		if d := parR[i] - serR[i]; d > 1e-9 || d < -1e-9 {
			t.Fatalf("IRFFT2 par/ser mismatch at %d: %v vs %v", i, parR[i], serR[i])
		}
	}
	// Real round trip under the parallel path.
	withWorkers(4, func() {
		back := IRFFT2(RFFT2(rx, [2]int{rows, cols}), [2]int{rows, cols})
		for i := range back {
			if d := back[i] - rx[i]; d > 1e-9 || d < -1e-9 {
				t.Fatalf("RFFT2 round trip at %d: %v vs %v", i, back[i], rx[i])
			}
		}
	})
}
