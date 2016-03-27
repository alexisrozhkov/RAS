## RAS  
Conversion of robust algebraic segmentation from Matlab to C++  
  
#### Paper:  
Robust Algebraic Segmentation of Mixed Rigid-Body and Planar Motions from Two Views  
  
#### Reference implementation:  
http://perception.csl.illinois.edu/ras/  
  
#### Dependencies:  
 - OpenCV 3.0  
 - Armadillo  
 - Google Test 1.7 (this particular revision seems to be the last one, which can be built with MinGW without much hassle)  
  
  
## Installation & building  
  
#### Ubuntu
1. Install OpenCV 3.0 and Armadillo  
2. `git clone https://github.com/alexisrozhkov/RAS.git`  
3. Download [Google Test 1.7](https://github.com/google/googletest/releases/tag/release-1.7.0)  
4. Unpack it to REPO_ROOT/tests/googletest  
5. `cd REPO_ROOT`  
6. `mkdir build && cd build`  
7. `cmake ..`  
8. `make`   
  
#### Windows (64 bit) using MinGW-w64 and OpenBLAS as Armadillo backend  
MinGW-w64 is used here, since vanilla MinGW has problems (like to_string is not a part of std, and other compatibility issues)  
  
1. Download OpenCV 3.0 sources and build them using MinGW-w64  
2. Download OpenBLAS sources and build them using MinGW-w64  
3. Download Armadillo (I haven't found a way to build it normally and discover using FindArmadillo on Windows, so following steps will be a bit hacky)  
  1. Browse to your CMake installation dir and locate FindArmadillo.cmake  
  2. Hardcode the path to Armadillo includes and to built OpenBLAS dll there  
4. `git clone https://github.com/alexisrozhkov/RAS.git`  
5. Download [Google Test 1.7](https://github.com/google/googletest/releases/tag/release-1.7.0)  
6. Unpack it to REPO_ROOT/tests/googletest  
7. `cd REPO_ROOT`  
8. `mkdir build && cd build`  
9. `cmake .. -G "MinGW Makefiles"`  
10. `mingw32-make`  