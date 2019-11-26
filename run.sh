#!/bin/bash
nvcc -o xpic xpic.cu
grep -v '#' init | ./xpic
