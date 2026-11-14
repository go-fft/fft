package fft

import "sync"

// This file implements the real two-dimensional transforms for image-style
// workloads, mirroring numpy.fft.rfft2 / irfft2.
//
// numpy.fft.rfft2(a) is fftn restricted to a real input where the LAST axis uses
// the real transform (keeping cols/2+1 non-redundant bins) and the remaining
// axis uses the full complex transform. irfft2 inverts that: the full complex
// inverse along the non-last axis, then the real inverse (irfft) along the last
// axis. The forward transform is unnormalized; irfft2 normalizes by the product
// of the reconstructed axis lengths. Inputs are never mutated.

// RFFT2 returns the forward 2-D DFT of a real row-major matrix of the given
// shape (shape[0] rows × shape[1] columns). The real transform is applied along
// the last axis, so each output row keeps shape[1]/2+1 non-redundant bins; the
// result is a row-major matrix of shape shape[0]×(shape[1]/2+1). The full
// complex transform is then applied down the columns. This matches
// numpy.fft.rfft2.
//
// shape lengths must be positive and shape[0]*shape[1] must equal len(data);
// RFFT2 panics otherwise. The input is not modified.
func RFFT2(data []float64, shape [2]int) []complex128 { return rfft2A(data, shape, arithDefault) }

// rfft2A is RFFT2 on a plan of arithmetic a.
func rfft2A(data []float64, shape [2]int, a arith) []complex128 {
	rows, cols := shape[0], shape[1]
	if shapeProduct(rows, cols) != len(data) {
		panic("fft: shape product does not match len(data)")
	}
	p := cachedRealPlan2A(rows, cols, a)
	return p.RFFT(make([]complex128, p.SpectrumLen()), data)
}

// IRFFT2 inverts RFFT2, reconstructing a real row-major matrix of shape
// shape[0]×shape[1] from a spectrum laid out as shape[0]×(shape[1]/2+1) complex
// bins (the layout RFFT2 produces). The complex inverse is applied down the
// columns first, then the real inverse (irfft) along each row, with the target
// row length shape[1] supplied explicitly (since shape[1] and shape[1]-1 share a
// bin count). The result is normalized by shape[0]*shape[1] so that
// IRFFT2(RFFT2(x, shape), shape) ≈ x. This matches numpy.fft.irfft2.
//
// shape lengths must be positive; IRFFT2 panics otherwise. data is read up to
// shape[0]*(shape[1]/2+1) bins; any beyond that are treated as zero. The input
// is not modified. The output has shape[0]*shape[1] values whatever the
// spectrum's length: bound the shape when it comes from untrusted input (see
// SECURITY.md).
func IRFFT2(data []complex128, shape [2]int) []float64 { return irfft2A(data, shape, arithDefault) }

// irfft2A is IRFFT2 on a plan of arithmetic a.
func irfft2A(data []complex128, shape [2]int, a arith) []float64 {
	rows, cols := shape[0], shape[1]
	shapeProduct(rows, cols)
	p := cachedRealPlan2A(rows, cols, a)
	spec := data
	if len(spec) != p.SpectrumLen() {
		// Fewer bins than the layout holds read as zeros, extra bins are
		// ignored: pad or trim to exactly the layout.
		spec = make([]complex128, p.SpectrumLen())
		copy(spec, data)
	}
	return p.IRFFT(make([]float64, rows*cols), spec)
}

// realPlan2Cache memoizes real 2-D plans by shape for RFFT2/IRFFT2.
var (
	realPlan2Mu    sync.Mutex
	realPlan2Cache = map[[2]int]*RealPlan2{}
	// realPlan2CacheFMA holds the FMA plans (cachedRealPlan2F).
	realPlan2CacheFMA = map[[2]int]*RealPlan2{}
	// realPlan2CacheExact holds the Exact plans (cachedRealPlan2A).
	realPlan2CacheExact = map[[2]int]*RealPlan2{}
)

// cachedRealPlan2 returns the shared plan for rows×cols, building it on first
// use under the lock (NewRealPlan2 takes only the 1-D caches' own mutexes).
func cachedRealPlan2(rows, cols int) *RealPlan2 { return cachedRealPlan2F(rows, cols, false) }

// cachedRealPlan2F is cachedRealPlan2 for an FMA plan when fma is set, from
// a cache of its own.
func cachedRealPlan2F(rows, cols int, fma bool) *RealPlan2 {
	return cachedRealPlan2A(rows, cols, arithFromFMA(fma))
}

// cachedRealPlan2A is cachedRealPlan2 for arithmetic a, from a cache of its own.
func cachedRealPlan2A(rows, cols int, a arith) *RealPlan2 {
	cache := realPlan2Cache
	switch a {
	case arithFMA:
		cache = realPlan2CacheFMA
	case arithExact:
		cache = realPlan2CacheExact
	}
	realPlan2Mu.Lock()
	defer realPlan2Mu.Unlock()
	key := [2]int{rows, cols}
	p, ok := cache[key]
	if !ok {
		p = newRealPlan2A(rows, cols, a)
		cache[key] = p
	}
	return p
}
