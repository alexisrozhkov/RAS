include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(runBasicRASTests
               basic_check.cpp)

target_link_libraries(runBasicRASTests gtest gtest_main)
