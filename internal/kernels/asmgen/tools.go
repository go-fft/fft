//go:build tools

// Package asmgen pins the go-asmgen version the arm64, riscv64 and s390x
// generators (*/gen.go) run with. Those carry //go:build ignore, so without
// this file go mod tidy saw no import of go-asmgen, dropped the requirement,
// and every `go run gen.go` resolved whatever go-asmgen release was newest
// that day: the committed assembly was only reproducible by luck. The amd64
// generator is its own module (amd64/go.mod) for the same reason.
package asmgen

import (
	_ "github.com/go-asmgen/asmgen/arm64"
	_ "github.com/go-asmgen/asmgen/emit"
	_ "github.com/go-asmgen/asmgen/riscv64"
	_ "github.com/go-asmgen/asmgen/s390x"
)
