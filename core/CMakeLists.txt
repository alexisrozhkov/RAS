cmake_minimum_required(VERSION 2.8)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${ARMADILLO_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR})

add_library(RAS STATIC
            utils/ras_types.cpp
            utils/mat_nd.cpp
            utils/balls_and_bins.cpp
            utils/math.cpp
            utils/subspace_angle.cpp
            utils/arma_wrapper.cpp
            utils/ras_params.cpp
            perspective_embedding.cpp
            find_polynomials.cpp
            robust_algebraic_segmentation.cpp)

target_link_libraries(RAS
                      ${OpenCV_LIBS}
                      ${ARMADILLO_LIBRARIES})
