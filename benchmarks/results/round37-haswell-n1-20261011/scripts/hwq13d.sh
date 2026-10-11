#!/bin/sh
cd ~/r37
./hwab.sh 10 1 whole-15.txt 2 -test.run '^TestHwWhole$' -hw.whole 15,6 -hw.shapes 8x1024,64x1024,256x1024,512x1024,1024x1024,4x256x1024,1024x512,256x512 -- br3-amd64
./hwab.sh 10 1 e11-ab2-15.txt 2 -test.run '^TestR26AB$' -r26.ab 15,100 -r26.sets '6144:rule,3x4x8x8x8;10240:rule,5x4x8x8x8;12288:rule;4096:rule;8192:rule' -- br3-amd64
echo "# timing done $(date +%T)" >> hwq13d.log
./br3-amd64.test > suite-fft.txt 2>&1; echo "fft exit $?" >> hwq13d.log
./brk3-amd64.test > suite-kernels.txt 2>&1; echo "kernels exit $?" >> hwq13d.log
echo "# queue d done $(date +%T)" >> hwq13d.log
