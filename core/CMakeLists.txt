cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CORE_FILES
        utils/balls_and_bins.cpp
        perspective_embedding.cpp)

add_library(RAS STATIC ${CORE_FILES})
target_link_libraries(RAS ${OpenCV_LIBS})
