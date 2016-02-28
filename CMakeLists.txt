cmake_minimum_required(VERSION 2.8)
project(RAS)

add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated")

add_subdirectory(core)
add_subdirectory(tests)

include_directories(core)
include_directories(core/utils)

set(SOURCE_FILES main.cpp)

add_executable(runRAS ${SOURCE_FILES})
target_link_libraries(runRAS RAS)
