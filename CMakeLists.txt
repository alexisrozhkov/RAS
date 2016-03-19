cmake_minimum_required(VERSION 2.8)
project(RAS)

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated -std=c++11")

add_subdirectory(core)
add_subdirectory(tests)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_executable(runRAS main.cpp)

target_link_libraries(runRAS ${OpenCV_LIBS} RAS)
