//go:build race

package fft

// raceEnabled: under -race sync.Pool drops items on purpose, so allocation
// counts are not meaningful.
const raceEnabled = true
