//go:build amd64

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// forceTableRoute keeps the generated 32- and 64-point kernels off for the test.
func forceTableRoute(tb testing.TB) {
	tb.Helper()
	old := kernels.Dct23On
	kernels.Dct23On = false
	tb.Cleanup(func() { kernels.Dct23On = old })
}
