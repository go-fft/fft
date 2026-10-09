#!/bin/sh
# ship.sh NAME [dir]: build the fft test binary from dir (default fft), ship to cfarm151:~/r31/NAME
set -e
S=/private/tmp/claude-501/-Users-david-delavennat-Documents-VCS-GIT-localhost/d368f442-d9da-4154-83cf-8b6b883bac13/scratchpad
KH=$S/kh-cascade31
NAME=$1; DIR=${2:-$S/agent-cascade31/fft}
cd $DIR
GOOS=linux GOARCH=amd64 go test -c -ldflags='-s -w' -o $S/agent-cascade31/bin/$NAME .
gzip -kf $S/agent-cascade31/bin/$NAME
L=$(stat -f %z $S/agent-cascade31/bin/$NAME.gz)
scp -q -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=yes -o UpdateHostKeys=no $S/agent-cascade31/bin/$NAME.gz delavennat@cfarm151.cfarm.net:r31/
R=$(ssh -x -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=yes -o UpdateHostKeys=no delavennat@cfarm151.cfarm.net "stat -c %s r31/$NAME.gz")
[ "$L" = "$R" ] || { echo "SIZE MISMATCH $L $R"; exit 1; }
ssh -x -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=yes -o UpdateHostKeys=no delavennat@cfarm151.cfarm.net "cd r31 && gunzip -f $NAME.gz && chmod +x $NAME && ls -l $NAME"
