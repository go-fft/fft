//go:build amd64 && !amd64.v3

package fft

// scalarFuses reports whether gc may fuse the scalar passes' multiply-adds.
// On amd64 below GOAMD64=v3 it never does, so the batched passes and the
// line-by-line ones agree bit for bit.
const scalarFuses = false
