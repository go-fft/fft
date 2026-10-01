#!/bin/sh
# Usage: setup.sh <fftw configure SIMD flags...>
set -e
cd ~/gofft-bench
if [ ! -d fftw-3.3.10 ]; then
  curl -sSLO https://www.fftw.org/fftw-3.3.10.tar.gz
  echo "56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467  fftw-3.3.10.tar.gz" | sha256sum -c
  tar xzf fftw-3.3.10.tar.gz
fi
if [ ! -f fftw/lib/libfftw3.a ]; then
  cd fftw-3.3.10
  ./configure --prefix="$HOME/gofft-bench/fftw" CFLAGS=-O3 "$@" > ../fftw-configure.log
  make -j16 > ../fftw-make.log 2>&1
  make install > /dev/null
  cd ..
fi
echo "FFTW-OK $(ls fftw/lib/libfftw3.a)"
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/pip -q install numpy scipy
echo "PY-OK $(venv/bin/python -c 'import numpy,scipy;print(numpy.__version__,scipy.__version__)')"
if venv/bin/pip -q install pyfftw 2>/dev/null; then echo "PYFFTW-OK"; else echo "PYFFTW-SKIP"; fi
