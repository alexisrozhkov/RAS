cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(utils)
include_directories(../3rdparty)

set(CORE_FILES
    utils/ras_types.cpp
    utils/balls_and_bins.cpp
    utils/generalized_eigenvalues.cpp
    perspective_embedding.cpp
    find_polynomials.cpp)

add_library(RAS STATIC ${CORE_FILES})
target_link_libraries(RAS ${OpenCV_LIBS})
