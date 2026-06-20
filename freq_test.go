package fft

import (
	"math"
	"testing"
)

// TestFFTFreqExact asserts the exact numpy.fft.fftfreq values for even and odd
// n, with both unit and non-unit spacing.
func TestFFTFreqExact(t *testing.T) {
	// numpy.fft.fftfreq(8, d=1) = [0, .125, .25, .375, -.5, -.375, -.25, -.125].
	want8 := []float64{0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125}
	closeReal(t, FFTFreq(8, 1), want8, "FFTFreq(8,1)")

	// numpy.fft.fftfreq(5, d=1) = [0, .2, .4, -.4, -.2].
	want5 := []float64{0, 0.2, 0.4, -0.4, -0.2}
	closeReal(t, FFTFreq(5, 1), want5, "FFTFreq(5,1)")

	// Non-unit spacing scales by 1/(d*n). numpy.fft.fftfreq(4, d=0.5):
	// 1/(0.5*4)=0.5 -> [0, 0.5, -1.0, -0.5].
	want4 := []float64{0, 0.5, -1.0, -0.5}
	closeReal(t, FFTFreq(4, 0.5), want4, "FFTFreq(4,0.5)")

	// n=1 -> [0].
	closeReal(t, FFTFreq(1, 1), []float64{0}, "FFTFreq(1,1)")
}

// TestRFFTFreqExact asserts the exact numpy.fft.rfftfreq values.
func TestRFFTFreqExact(t *testing.T) {
	// numpy.fft.rfftfreq(8, d=1) = [0, .125, .25, .375, .5].
	closeReal(t, RFFTFreq(8, 1), []float64{0, 0.125, 0.25, 0.375, 0.5}, "RFFTFreq(8,1)")
	// numpy.fft.rfftfreq(5, d=1) = [0, .2, .4].
	closeReal(t, RFFTFreq(5, 1), []float64{0, 0.2, 0.4}, "RFFTFreq(5,1)")
	// Non-unit spacing: rfftfreq(4, d=0.25): 1/(0.25*4)=1 -> [0,1,2].
	closeReal(t, RFFTFreq(4, 0.25), []float64{0, 1, 2}, "RFFTFreq(4,0.25)")
}

func TestFreqEdgeCases(t *testing.T) {
	if got := FFTFreq(0, 1); got == nil || len(got) != 0 {
		t.Fatalf("FFTFreq(0): got %v", got)
	}
	if got := FFTFreq(-2, 1); got == nil || len(got) != 0 {
		t.Fatalf("FFTFreq(-2): got %v", got)
	}
	if got := RFFTFreq(0, 1); got == nil || len(got) != 0 {
		t.Fatalf("RFFTFreq(0): got %v", got)
	}
	if got := RFFTFreq(-5, 1); got == nil || len(got) != 0 {
		t.Fatalf("RFFTFreq(-5): got %v", got)
	}
}

// TestFFTFreqLengthAndOrder checks structural invariants for several n: the
// length is n, the DC bin is 0, the positive half increases, and the result
// matches the numpy ordering convention against a brute reference.
func TestFFTFreqMatchesReference(t *testing.T) {
	ref := func(n int, d float64) []float64 {
		out := make([]float64, n)
		val := 1.0 / (d * float64(n))
		for i := 0; i < n; i++ {
			k := i
			if i > (n-1)/2 {
				k = i - n // wrap to negative frequency
			}
			out[i] = float64(k) * val
		}
		return out
	}
	for _, n := range []int{1, 2, 3, 6, 7, 10, 11} {
		closeReal(t, FFTFreq(n, 0.3), ref(n, 0.3), "FFTFreq vs ref")
	}
}

// TestPSDParseval ties the one-sided PSD back to signal energy: summing the
// density over the frequency bins (times the bin width) recovers the mean square
// of the signal (Parseval, density scaling). This validates PSD independent of
// the FFT internals.
func TestPSDParseval(t *testing.T) {
	n := 32
	d := 0.1
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(2*math.Pi*3*float64(i)/float64(n)) + 0.5*math.Cos(2*math.Pi*7*float64(i)/float64(n))
	}
	psd := PSD(x, d)
	freqs := RFFTFreq(n, d)
	if len(psd) != n/2+1 {
		t.Fatalf("PSD length %d, want %d", len(psd), n/2+1)
	}
	// Integrate density over frequency: sum(psd) * df, df = freqs[1]-freqs[0].
	df := freqs[1] - freqs[0]
	var integral float64
	for _, p := range psd {
		integral += p * df
	}
	var meanSq float64
	for _, v := range x {
		meanSq += v * v
	}
	meanSq /= float64(n)
	if d := math.Abs(integral - meanSq); d > 1e-9 {
		t.Fatalf("PSD integral %.12g != mean square %.12g (|d|=%g)", integral, meanSq, d)
	}
}

// TestPSDOddNyquistDoubling exercises the odd-N branch (no Nyquist bin to leave
// undoubled) via Parseval again on an odd length.
func TestPSDOddParseval(t *testing.T) {
	n := 15
	d := 1.0
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Cos(2*math.Pi*2*float64(i)/float64(n)) + float64(i%3)
	}
	psd := PSD(x, d)
	freqs := RFFTFreq(n, d)
	df := freqs[1] - freqs[0]
	var integral float64
	for _, p := range psd {
		integral += p * df
	}
	var meanSq float64
	for _, v := range x {
		meanSq += v * v
	}
	meanSq /= float64(n)
	if dd := math.Abs(integral - meanSq); dd > 1e-9 {
		t.Fatalf("odd PSD integral %.12g != mean square %.12g", integral, meanSq)
	}
}

func TestPSDEmpty(t *testing.T) {
	if got := PSD([]float64{}, 1); got == nil || len(got) != 0 {
		t.Fatalf("PSD empty: got %v", got)
	}
}

func TestPSDNoMutation(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	orig := make([]float64, len(x))
	copy(orig, x)
	PSD(x, 1)
	for i := range x {
		if x[i] != orig[i] {
			t.Fatalf("PSD mutated input at %d", i)
		}
	}
}

func TestSpectrogramFrames(t *testing.T) {
	x := make([]float64, 20)
	for i := range x {
		x[i] = math.Sin(float64(i))
	}
	segment, overlap := 8, 4
	win := Hann(segment)
	frames := Spectrogram(x, segment, overlap, win, 1.0)
	// step = 4; starts at 0,4,8,12 (12+8=20 fits); 16+8=24 does not.
	if len(frames) != 4 {
		t.Fatalf("frame count %d, want 4", len(frames))
	}
	for i, f := range frames {
		if len(f) != segment/2+1 {
			t.Fatalf("frame %d length %d, want %d", i, len(f), segment/2+1)
		}
	}

	// Each frame must equal PSD of its windowed segment, computed independently.
	for t0, start := 0, 0; t0 < len(frames); t0, start = t0+1, start+4 {
		seg := make([]float64, segment)
		for i := 0; i < segment; i++ {
			seg[i] = x[start+i] * win[i]
		}
		want := PSD(seg, 1.0)
		closeReal(t, frames[t0], want, "spectrogram frame")
	}
}

// TestSpectrogramNoFullSegment confirms an input shorter than one segment yields
// no frames (nil result).
func TestSpectrogramNoFullSegment(t *testing.T) {
	x := []float64{1, 2, 3}
	frames := Spectrogram(x, 8, 0, Hann(8), 1.0)
	if frames != nil {
		t.Fatalf("expected nil frames, got %v", frames)
	}
}

func TestSpectrogramPanics(t *testing.T) {
	cases := []struct {
		fn    func()
		label string
	}{
		{func() { Spectrogram(nil, 0, 0, nil, 1) }, "segment<=0"},
		{func() { Spectrogram(nil, 4, 0, Hann(3), 1) }, "window length mismatch"},
		{func() { Spectrogram(nil, 4, -1, Hann(4), 1) }, "overlap negative"},
		{func() { Spectrogram(nil, 4, 4, Hann(4), 1) }, "overlap >= segment"},
	}
	for _, c := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s: expected panic", c.label)
				}
			}()
			c.fn()
		}()
	}
}
