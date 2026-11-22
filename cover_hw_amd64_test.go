//go:build amd64

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// Wide is read by the frame tests only on CPUs whose plans take a wide final pass;
// a zero plan answers it on any CPU, so the gate does not depend on the host.
func TestZnPlanWideZeroValue(t *testing.T) {
	if (&kernels.ZnPlan{}).Wide() {
		t.Fatal("a zero ZnPlan reports a wide final pass")
	}
}
