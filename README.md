## RAS  
Conversion of robust algebraic segmentation from Matlab to C++  
  
#### Paper:  
Robust Algebraic Segmentation of Mixed Rigid-Body and Planar Motions from Two Views  
  
#### Reference implementation:  
http://perception.csl.illinois.edu/ras/  
  
#### Dependencies:  
 - OpenCV 3.0  
 - Armadillo with OpenBLAS as backend. I have used Armadillo 6.600.5 "Catabolic Amalgamator" and OpenBLAS 0.2.18.dev  
 - Google Test 1.7 (this particular revision seems to be the last one, which can be built with MinGW without much hassle)  
  
  
## Installation & building  
On Windows MinGW-w64 is used for building, since vanilla MinGW has problems (like to_string is not a part of std, and other compatibility issues)  
  
1. Download OpenCV 3.0, build (use MinGW-w64 on Windows) and install it  
2. Download OpenBLAS sources and build them (use MinGW-w64 on Windows), path to the resulting binary will be referred as PATH_TO_LIBOPENBLAS_BIN  
3. Download Armadillo sources, path to its includes will be referred as PATH_TO_ARMADILLO_INCLUDES  
4. `git clone https://github.com/alexisrozhkov/RAS.git`, resulting directory will be referred as REPO_ROOT  
5. Download [Google Test 1.7](https://github.com/google/googletest/releases/tag/release-1.7.0)  
6. Unpack it to REPO_ROOT/tests/googletest  
7. `cd REPO_ROOT`  
8. `mkdir build && cd build`  
9. `cmake .. -DARMADILLO_INCLUDE_DIRS="PATH_TO_ARMADILLO_INCLUDES" -DARMADILLO_LIBRARIES="PATH_TO_LIBOPENBLAS_BIN"`, replace PATH_TO_ARMADILLO_INCLUDES and PATH_TO_LIBOPENBLAS_BIN with your values, add `-G "MinGW Makefiles"` on Windows  
10. `make` or `mingw32-make` on Windows  
