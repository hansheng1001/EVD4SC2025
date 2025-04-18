# this is open code for SC25's paper "Rethinking Back Transformation in 2-Stage Eigenvalue Decomposition on Heterogeneous Architectures"

# build
cmake -B bulid -S .
cmake --build build/ -j

# test
./bulid/src/EVD/myEVD {n} 32 1024 (e.g. ./bulid/src/EVD/myEVD 16384 32 1024) 

