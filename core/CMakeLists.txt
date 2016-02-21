cmake_minimum_required(VERSION 2.8)

set(CORE_FILES utils/balls_and_bins.cpp)

add_library(RAS STATIC ${CORE_FILES})
