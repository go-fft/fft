package fft

import (
	"runtime"
	"sync"
	"testing"
)

// chunkCount returns how many times parChunks calls body for n items of
// itemLen elements each.
func chunkCount(n, itemLen int) int {
	var mu sync.Mutex
	calls := 0
	parChunks(n, itemLen, func(lo, hi int) {
		mu.Lock()
		calls++
		mu.Unlock()
	})
	return calls
}

// TestParWorkersFollowsGOMAXPROCS pins issue #107: the worker count is read
// when a transform runs, not once at package initialisation, so lowering
// GOMAXPROCS afterwards (runtime.GOMAXPROCS(1), go test -cpu 1) is honoured.
func TestParWorkersFollowsGOMAXPROCS(t *testing.T) {
	if parWorkers != 0 {
		t.Fatalf("parWorkers = %d outside a test override, want 0 (read GOMAXPROCS per call)", parWorkers)
	}
	prev := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(prev)

	for _, procs := range []int{1, 2, 3} {
		runtime.GOMAXPROCS(procs)
		if got := parWorkerCount(); got != procs {
			t.Fatalf("GOMAXPROCS(%d): parWorkerCount() = %d", procs, got)
		}
		// Plenty of items and plenty of work per item: one chunk per worker.
		if got := chunkCount(64, 10*parMinChunk); got != procs {
			t.Errorf("GOMAXPROCS(%d): parChunks ran %d chunks, want %d", procs, got, procs)
		}
		if got, want := parallelizeLines(1000, 1000), procs > 1; got != want {
			t.Errorf("GOMAXPROCS(%d): parallelizeLines = %v, want %v", procs, got, want)
		}
		if got, want := parallelizeAxis(1000, 1000), procs > 1; got != want {
			t.Errorf("GOMAXPROCS(%d): parallelizeAxis = %v, want %v", procs, got, want)
		}
	}
}

// TestPlanNOneCoreAllocatesNothing is the witness of #107: with GOMAXPROCS
// lowered after init, PlanN.FFT used to spawn goroutines (36 allocations per
// call on a 256x256 matrix). On one core it runs inline and, in steady state,
// allocates nothing, as the PlanN documentation says.
func TestPlanNOneCoreAllocatesNothing(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	shape := []int{256, 256}
	p := NewPlanN(shape...)
	src := complexGrid(shape)
	dst := make([]complex128, len(src))
	p.FFT(dst, src)
	if got := testing.AllocsPerRun(10, func() { p.FFT(dst, src) }); got != 0 {
		t.Fatalf("PlanN.FFT 256x256 on GOMAXPROCS(1): %v allocations per call, want 0", got)
	}
}
