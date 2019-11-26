#!/bin/bash
rm xpic
nvcc -o xpic xpic.cu
grep -v '#' init | ./xpic
