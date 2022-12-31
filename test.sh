#!/bin/bash

for size in "16 16 16" "100 100 100" "128 128 128" "256 256 128"; do
  echo -e "\nSize: $size"
  mpirun -np 1 ./build/test $size
done