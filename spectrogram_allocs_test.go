package fft

import (
	"math"
	"testing"
)

// spectrogramFrames is the per-frame definition Spectrogram is documented
// by: window the segment, then PSD. Spectrogram must match it bit for bit.
func spectrogramFrames(x []float64, segment, overlap int, window []float64, d float64) [][]float64 {
	var frames [][]float64
	seg := make([]float64, segment)
	for start := 0; start+segment <= len(x); start += segment - overlap {
		for i := range seg {
			seg[i] = x[start+i] * window[i]
		}
		frames = append(frames, PSD(seg, d))
	}
	return frames
}

func TestSpectrogramBitIdenticalToPSDFrames(t *testing.T) {
	for _, segment := range []int{1, 2, 7, 64, 255, 256} {
		x := make([]float64, 4096+3)
		for i := range x {
			x[i] = math.Sin(0.37*float64(i)) + 0.25*math.Cos(0.011*float64(i*i%977))
		}
		w := make([]float64, segment)
		for i := range w {
			w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(segment))
		}
		for _, overlap := range []int{0, segment / 2, segment - 1} {
			got := Spectrogram(x, segment, overlap, w, 1.0/48000)
			want := spectrogramFrames(x, segment, overlap, w, 1.0/48000)
			if len(got) != len(want) {
				t.Fatalf("segment %d overlap %d: %d frames, want %d", segment, overlap, len(got), len(want))
			}
			for f := range got {
				if len(got[f]) != len(want[f]) {
					t.Fatalf("frame %d length %d, want %d", f, len(got[f]), len(want[f]))
				}
				for k := range got[f] {
					if math.Float64bits(got[f][k]) != math.Float64bits(want[f][k]) {
						t.Fatalf("segment %d overlap %d frame %d bin %d: %v, want %v", segment, overlap, f, k, got[f][k], want[f][k])
					}
				}
			}
		}
	}
}

// TestSpectrogramAllocs pins the allocation count: a fixed handful per call
// (windowed segment, spectrum scratch, one backing array, the frame list),
// whatever the frame count. It was 2 per frame plus append growth (262 at 127
// frames).
func TestSpectrogramAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("under -race sync.Pool drops items on purpose, so allocation counts are not meaningful")
	}
	x := make([]float64, 16384)
	for i := range x {
		x[i] = math.Sin(0.1 * float64(i))
	}
	w := make([]float64, 256)
	for i := range w {
		w[i] = 1
	}
	Spectrogram(x, 256, 128, w, 1) // warm the plan cache
	got := testing.AllocsPerRun(20, func() { Spectrogram(x, 256, 128, w, 1) })
	if got != 4 {
		t.Fatalf("Spectrogram: %v allocations per call, want 4", got)
	}
	long := make([]float64, 4*len(x))
	if g := testing.AllocsPerRun(20, func() { Spectrogram(long, 256, 128, w, 1) }); g != 4 {
		t.Fatalf("Spectrogram (4x the frames): %v allocations per call, want 4", g)
	}
}
