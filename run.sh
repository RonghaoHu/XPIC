#!/bin/bash
rm xpic
rm *.h5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/hdf5/serial/
nvcc -I/usr/include/hdf5/serial/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5 -o xpic xpic.cu
grep -v '#' init | ./xpic
