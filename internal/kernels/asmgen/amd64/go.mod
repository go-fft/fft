// The amd64 kernel generator is its own module so the go-asmgen version it
// generates with is pinned here, not resolved from whatever a cache holds:
// the committed .s files are reproducible only if the generator is.
module github.com/go-fft/fft/internal/kernels/asmgen/amd64

go 1.27.1

require github.com/go-asmgen/asmgen v0.17.0

replace github.com/go-asmgen/asmgen => /private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent-r42/asmgen
