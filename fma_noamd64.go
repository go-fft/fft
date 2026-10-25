//go:build !amd64

package fft

import "github.com/go-fft/fft/internal/kernels"

// fmaSplitDefault: there are no FMA kernels here, so a split pass keeps its
// kernel (arm64's NEON ones) rather than its Go fallback.
func fmaSplitDefault() bool { return false }

// fmaTableDefault is nil: FMA plans here are built by the tests only.
func fmaTableDefault() map[int]fmaChoice { return nil }

// znFMABind is the frame plans' binder for an FMA plan: there are no FMA
// frame kernels here, so an FMA plan binds what a default one does
// (kernels.ZnPlanForWide, nil where there are no frame kernels at all).
func znFMABind([]skStage) znBinder { return kernels.ZnPlanForWide }
