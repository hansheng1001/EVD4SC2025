# this is open code for APSLOS25 syEVD on GPU

# build
cmake -B bulid -S .
cmake --build .   

# test
./bulid/src/my_SB2TR_ZY_ZY_V3 32768 32 1024 
