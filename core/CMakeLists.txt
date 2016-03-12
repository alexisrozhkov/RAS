cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${ARMADILLO_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR})

add_library(RAS STATIC
            utils/ras_types.cpp
            utils/mat_nd.cpp
            utils/balls_and_bins.cpp
            utils/generalized_eigenvalues.cpp
            utils/cholesky.cpp
            utils/subspace_angle.cpp
            utils/jacobi_svd.cpp
            utils/arma_svd.cpp
            perspective_embedding.cpp
            find_polynomials.cpp
            robust_algebraic_segmentation.cpp)

target_link_libraries(RAS
                      ${OpenCV_LIBS}
                      ${ARMADILLO_LIBRARIES})
