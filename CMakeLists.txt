cmake_minimum_required(VERSION 2.8)
project(runRAS)

add_definitions(-std=c++11)

include_directories(core/utils)
set(SOURCE_FILES main.cpp)

add_subdirectory(core)
add_subdirectory(tests)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} RAS)
