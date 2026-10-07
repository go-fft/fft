#!/bin/sh
# muts.sh: each generator mutant against the per-pass and the whole-transform tests.
cd ~/gofft-bench/r28 && mkdir -p mut && tar xzf muts.tgz -C mut && cp -r testdata mut/
cd mut
echo "# start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
for b in m_*.test; do
  m=${b#m_}; m=${m%.test}
  p=$(./$b -test.run '^TestIntelSplit512EachPassMatchesScalar$' 2>&1 | grep -m1 -E 'intelsplit512.*:|^(ok|PASS)$|panic|SIG' )
  w=$(./$b -test.run '^TestIntelSplit512TransformMatchesScalar$' 2>&1 | grep -m1 -E 'intelsplit512.*:|^(ok|PASS)$|panic|SIG' )
  a=$(./$b -test.run 'TestStockhamPassMatchesScalar|TestSplitTransform|TestFFT' 2>&1 | grep -m1 -E '_test.go:[0-9]+:|^(ok|PASS)$|panic|SIG' )
  echo "== $m per-pass: $p"
  echo "== $m transform: $w"
  echo "== $m other suites: $a"
done
echo "# end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
