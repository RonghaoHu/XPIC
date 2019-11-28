#!/bin/bash
rm xpic
nvcc -DHIGH_PRECISION -o xpic xpic.cu
grep -v '#' init | ./xpic
