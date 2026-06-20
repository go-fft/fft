package fft

// This file provides the bin-frequency helpers and convenience spectral
// estimators, mirroring numpy.fft.fftfreq / rfftfreq and the textbook
// periodogram / spectrogram definitions.

// FFTFreq returns the n sample frequencies of an FFT of length n, given the
// sample spacing d (the inverse of the sampling rate). The result matches
// numpy.fft.fftfreq(n, d) exactly:
//
//	f = [0, 1, ..., (n-1)//2, -(n//2), ..., -1] / (d·n).
//
// The DC bin is first, then the positive frequencies in increasing order, then
// the negative frequencies. For even n the Nyquist bin appears as -n/2; for odd
// n the largest positive bin is (n-1)/2. n <= 0 returns an empty (non-nil)
// slice.
func FFTFreq(n int, d float64) []float64 {
	if n <= 0 {
		return []float64{}
	}
	out := make([]float64, n)
	val := 1.0 / (d * float64(n))
	// Positive (and zero) half: indices 0 .. (n-1)/2.
	half := (n - 1) / 2
	for i := 0; i <= half; i++ {
		out[i] = float64(i) * val
	}
	// Negative half: -(n/2), ..., -1 placed at indices half+1 .. n-1.
	j := -(n / 2)
	for i := half + 1; i < n; i++ {
		out[i] = float64(j) * val
		j++
	}
	return out
}

// RFFTFreq returns the n/2+1 sample frequencies of a real FFT (RFFT) of input
// length n, given the sample spacing d. The result matches
// numpy.fft.rfftfreq(n, d) exactly:
//
//	f = [0, 1, ..., n//2] / (d·n),  length n//2+1.
//
// All frequencies are non-negative because RFFT keeps only the non-redundant
// lower half of the spectrum. n <= 0 returns an empty (non-nil) slice.
func RFFTFreq(n int, d float64) []float64 {
	if n <= 0 {
		return []float64{}
	}
	m := n/2 + 1
	out := make([]float64, m)
	val := 1.0 / (d * float64(n))
	for i := 0; i < m; i++ {
		out[i] = float64(i) * val
	}
	return out
}

// PSD returns the one-sided power spectral density (periodogram) of the
// real-valued signal x sampled at spacing d, using the non-redundant RFFT bins.
// The result has length len(x)/2+1.
//
// Each bin is scaled to a density: |X[k]|² / (fs·N) where fs = 1/d is the
// sampling rate and N = len(x). To conserve total power, every bin except DC
// (and the Nyquist bin for even N) is doubled to fold in the discarded mirror
// half. This matches scipy.signal.periodogram(x, fs, window="boxcar",
// scaling="density", return_onesided=True). The companion frequency axis is
// RFFTFreq(len(x), d). The input is not modified; an empty x returns an empty
// (non-nil) slice.
func PSD(x []float64, d float64) []float64 {
	n := len(x)
	if n == 0 {
		return []float64{}
	}
	spec := RFFT(x)
	m := len(spec)
	out := make([]float64, m)
	// Density scale: 1 / (fs·N) with fs = 1/d, i.e. d / N.
	scale := d / float64(n)
	for k := 0; k < m; k++ {
		re, im := real(spec[k]), imag(spec[k])
		p := (re*re + im*im) * scale
		// Double every bin except DC and (for even N) the Nyquist bin, to account
		// for the folded negative-frequency half.
		if k != 0 && !(n%2 == 0 && k == m-1) {
			p *= 2
		}
		out[k] = p
	}
	return out
}

// Spectrogram computes a sequence of one-sided power spectra over successive,
// optionally overlapping segments of the real signal x, the building block of a
// time–frequency display. segment is the segment length (window size) and
// overlap is the number of samples shared between consecutive segments
// (0 <= overlap < segment). Each segment is multiplied by window (which must
// have length segment) before its PSD is taken.
//
// The return value is a slice of frames, each a []float64 of length
// segment/2+1 (the RFFTFreq(segment, d) axis); frame t covers samples
// [t·step, t·step+segment) with step = segment-overlap. Segments are taken only
// while a full window fits, so trailing samples shorter than segment are
// dropped (the common STFT convention). The input is not modified.
//
// Spectrogram panics if segment <= 0, len(window) != segment, or overlap is out
// of range, mirroring the contract of scipy.signal.spectrogram.
func Spectrogram(x []float64, segment, overlap int, window []float64, d float64) [][]float64 {
	if segment <= 0 {
		panic("fft: segment length must be positive")
	}
	if len(window) != segment {
		panic("fft: window length must equal segment length")
	}
	if overlap < 0 || overlap >= segment {
		panic("fft: overlap must satisfy 0 <= overlap < segment")
	}
	step := segment - overlap
	var frames [][]float64
	seg := make([]float64, segment)
	for start := 0; start+segment <= len(x); start += step {
		for i := 0; i < segment; i++ {
			seg[i] = x[start+i] * window[i]
		}
		frames = append(frames, PSD(seg, d))
	}
	return frames
}
