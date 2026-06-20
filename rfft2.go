package fft

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
func RFFT2(data []float64, shape [2]int) []complex128 {
	rows, cols := shape[0], shape[1]
	if rows <= 0 || cols <= 0 {
		panic("fft: shape lengths must be positive")
	}
	if rows*cols != len(data) {
		panic("fft: shape product does not match len(data)")
	}

	rcols := cols/2 + 1

	// Step 1: real FFT along each row, producing a rows×rcols complex matrix.
	// One row plan is reused across all rows, and the independent rows run across
	// goroutines when the matrix is large enough.
	half := make([]complex128, rows*rcols)
	rp := cachedRealPlan(cols)
	rowWork := func(lo, hi int) {
		row := make([]float64, cols)
		dst := make([]complex128, rcols)
		for r := lo; r < hi; r++ {
			copy(row, data[r*cols:(r+1)*cols])
			rp.RFFT(dst, row)
			copy(half[r*rcols:(r+1)*rcols], dst)
		}
	}
	if parallelizeLines(rows, cols) {
		parChunks(rows, rowWork)
	} else {
		rowWork(0, rows)
	}

	// Step 2: full complex FFT down each column of the rows×rcols matrix.
	cp := cachedPlan(rows)
	colWork := func(lo, hi int) {
		col := make([]complex128, rows)
		dst := make([]complex128, rows)
		for c := lo; c < hi; c++ {
			for r := 0; r < rows; r++ {
				col[r] = half[r*rcols+c]
			}
			cp.FFT(dst, col)
			for r := 0; r < rows; r++ {
				half[r*rcols+c] = dst[r]
			}
		}
	}
	if parallelizeLines(rcols, rows) {
		parChunks(rcols, colWork)
	} else {
		colWork(0, rcols)
	}
	return half
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
// is not modified.
func IRFFT2(data []complex128, shape [2]int) []float64 {
	rows, cols := shape[0], shape[1]
	if rows <= 0 || cols <= 0 {
		panic("fft: shape lengths must be positive")
	}
	rcols := cols/2 + 1

	// Copy the supplied bins into a rows×rcols working matrix, zero-padding any
	// bins the caller did not provide.
	half := make([]complex128, rows*rcols)
	for i := 0; i < rows*rcols && i < len(data); i++ {
		half[i] = data[i]
	}

	// Step 1: complex inverse FFT down each column.
	cp := cachedPlan(rows)
	colWork := func(lo, hi int) {
		col := make([]complex128, rows)
		dst := make([]complex128, rows)
		for c := lo; c < hi; c++ {
			for r := 0; r < rows; r++ {
				col[r] = half[r*rcols+c]
			}
			cp.IFFT(dst, col)
			for r := 0; r < rows; r++ {
				half[r*rcols+c] = dst[r]
			}
		}
	}
	if parallelizeLines(rcols, rows) {
		parChunks(rcols, colWork)
	} else {
		colWork(0, rcols)
	}

	// Step 2: real inverse (irfft) along each row back to length cols.
	out := make([]float64, rows*cols)
	rp := cachedRealPlan(cols)
	rowWork := func(lo, hi int) {
		dst := make([]float64, cols)
		for r := lo; r < hi; r++ {
			rp.IRFFT(dst, half[r*rcols:(r+1)*rcols])
			copy(out[r*cols:(r+1)*cols], dst)
		}
	}
	if parallelizeLines(rows, cols) {
		parChunks(rows, rowWork)
	} else {
		rowWork(0, rows)
	}
	return out
}
