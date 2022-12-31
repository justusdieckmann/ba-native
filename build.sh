#! /bin/bash
rm -rf build
mkdir build
cd build || exit
cmake .. -DDOGRAPHICS=0
cmake --build . --target test