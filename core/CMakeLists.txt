cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ..)

add_library(RAS STATIC
            utils/ras_types.cpp
            utils/balls_and_bins.cpp
            utils/generalized_eigenvalues.cpp
            perspective_embedding.cpp
            find_polynomials.cpp)

target_link_libraries(RAS ${OpenCV_LIBS})
