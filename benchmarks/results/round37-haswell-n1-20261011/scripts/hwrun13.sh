#!/bin/sh
# hwrun13.sh PATTERN: build this tree's linux/amd64 fft test binary, run PATTERN on Haswell.
set -e
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -c -o /private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent-r37/mut13.test .
/private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent-r37/bin/hwscp13 /private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent-r37/mut13.test delavennat@cfarm13.cfarm.net:r37/mut13.test
/private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad/agent-r37/bin/hw13 "cd ~/r37 && ./mut13.test -test.run '$1'"
