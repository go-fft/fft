//go:build !amd64

package fft

import "testing"

// forceTableRoute: there is no kernel off amd64, the table route is the only one.
func forceTableRoute(tb testing.TB) { tb.Helper() }
