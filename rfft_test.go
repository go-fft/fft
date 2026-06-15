package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

// realSignal builds a deterministic, non-trivial real signal of length n.
func realSignal(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.7) + 0.3*math.Cos(float64(i)*1.9) + float64(i%4)
	}
	return x
}

// TestRFFTMatchesFullSpectrum checks RFFT against the kept slice of the full
// complex FFT (the naive-DFT path FFTReal is itself cross-checked elsewhere),
// across even, odd, power-of-two, and Bluestein lengths.
func TestRFFTMatchesFullSpectrum(t *testing.T) {
	for _, n := range []int{2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32} {
		x := realSignal(n)
		got := RFFT(x)
		full := FFTReal(x)
		want := full[:n/2+1]
		if len(got) != n/2+1 {
			t.Fatalf("n=%d: length %d, want %d", n, len(got), n/2+1)
		}
		closeVec(t, got, want)
	}
}

// TestRFFTConjugateSymmetry verifies the dropped upper half really is the
// conjugate mirror, justifying keeping only n/2+1 bins.
func TestRFFTConjugateSymmetry(t *testing.T) {
	n := 12
	x := realSignal(n)
	half := RFFT(x)
	full := FFTReal(x)
	for k := 1; k < n-k; k++ {
		if d := cmplx.Abs(full[n-k] - cmplx.Conj(half[k])); d > tol {
			t.Fatalf("bin %d: mirror mismatch |diff|=%g", k, d)
		}
	}
}

// TestRFFTDCAndNyquistAreReal checks that for even n the DC (bin 0) and Nyquist
// (bin n/2) bins of a real signal have zero imaginary part.
func TestRFFTDCAndNyquistAreReal(t *testing.T) {
	n := 8
	x := realSignal(n)
	X := RFFT(x)
	if math.Abs(imag(X[0])) > tol {
		t.Fatalf("DC bin not real: %v", X[0])
	}
	if math.Abs(imag(X[n/2])) > tol {
		t.Fatalf("Nyquist bin not real: %v", X[n/2])
	}
}

// TestIRFFTRoundTrip reconstructs the original real signal from its half
// spectrum for even and odd lengths.
func TestIRFFTRoundTrip(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 8, 9, 12, 15, 16, 17, 32} {
		x := realSignal(n)
		got := IRFFT(RFFT(x), n)
		if len(got) != n {
			t.Fatalf("n=%d: length %d, want %d", n, len(got), n)
		}
		for i := range got {
			if d := math.Abs(got[i] - x[i]); d > tol {
				t.Fatalf("n=%d index %d: got %g, want %g (|diff|=%g)", n, i, got[i], x[i], d)
			}
		}
	}
}

// TestRFFTKnownSinusoid uses an analytic signal: a single cosine at bin f over
// an even length puts all energy in bin f with magnitude n/2 (and the DC/Nyquist
// bins zero), independent of any reference implementation.
func TestRFFTKnownSinusoid(t *testing.T) {
	n := 16
	f := 3
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Cos(2 * math.Pi * float64(f) * float64(i) / float64(n))
	}
	X := RFFT(x)
	for k := range X {
		mag := cmplx.Abs(X[k])
		want := 0.0
		if k == f {
			want = float64(n) / 2
		}
		if math.Abs(mag-want) > tol {
			t.Fatalf("bin %d: magnitude %g, want %g", k, mag, want)
		}
	}
}

// TestIRFFTHermitianMirror feeds IRFFT a synthetic half spectrum and confirms
// the reconstructed signal matches the full inverse transform of the mirrored
// Hermitian spectrum (covers the conjugate-fill branch directly).
func TestIRFFTHermitianMirror(t *testing.T) {
	n := 6
	half := []complex128{complex(3, 0), complex(1, -2), complex(-0.5, 0.4), complex(2, 0)}
	got := IRFFT(half, n)

	// Build the expected full Hermitian spectrum by hand and invert it.
	full := make([]complex128, n)
	full[0] = half[0]
	full[1] = half[1]
	full[2] = half[2]
	full[3] = half[3]
	full[4] = cmplx.Conj(half[2])
	full[5] = cmplx.Conj(half[1])
	inv := IFFT(full)
	for i := 0; i < n; i++ {
		if d := math.Abs(got[i] - real(inv[i])); d > tol {
			t.Fatalf("index %d: got %g, want %g", i, got[i], real(inv[i]))
		}
	}
}

func TestRFFTEdgeCases(t *testing.T) {
	if got := RFFT([]float64{}); got == nil || len(got) != 0 {
		t.Fatalf("RFFT empty: got %v", got)
	}
	// Length 1: single real bin.
	if got := RFFT([]float64{5}); len(got) != 1 || got[0] != complex(5, 0) {
		t.Fatalf("RFFT len-1: got %v", got)
	}
	// IRFFT with non-positive n returns empty, non-nil.
	if got := IRFFT([]complex128{complex(1, 0)}, 0); got == nil || len(got) != 0 {
		t.Fatalf("IRFFT n=0: got %v", got)
	}
	if got := IRFFT([]complex128{complex(1, 0)}, -3); got == nil || len(got) != 0 {
		t.Fatalf("IRFFT n=-3: got %v", got)
	}
	// IRFFT length-1 spectrum, n=1: returns the real DC value.
	if got := IRFFT([]complex128{complex(4, 0)}, 1); len(got) != 1 || math.Abs(got[0]-4) > tol {
		t.Fatalf("IRFFT n=1: got %v", got)
	}
	// Short spectrum: missing bins treated as zero (covers the loop guard where
	// len(spectrum) < n/2+1).
	got := IRFFT([]complex128{complex(2, 0)}, 4)
	want := IRFFT([]complex128{complex(2, 0), 0, 0}, 4)
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("short spectrum index %d: got %g, want %g", i, got[i], want[i])
		}
	}
}

func TestRFFTNoMutation(t *testing.T) {
	n := 8
	x := realSignal(n)
	orig := make([]float64, n)
	copy(orig, x)
	RFFT(x)
	for i := range x {
		if x[i] != orig[i] {
			t.Fatalf("RFFT mutated input at %d", i)
		}
	}

	spec := RFFT(x)
	specOrig := make([]complex128, len(spec))
	copy(specOrig, spec)
	IRFFT(spec, n)
	for i := range spec {
		if spec[i] != specOrig[i] {
			t.Fatalf("IRFFT mutated input at %d", i)
		}
	}
}
