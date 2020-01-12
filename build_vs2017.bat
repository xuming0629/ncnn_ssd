mkdir -p build_vs2017
cd build_vs2017
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install ..
nmake
nmake install