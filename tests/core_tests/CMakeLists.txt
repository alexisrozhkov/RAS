include_directories(${gtest_SOURCE_DIR}/include
                    ${gtest_SOURCE_DIR}
                    ../../core
                    ../../core/utils)

add_executable(runCoreTests
               balls_and_bins.cpp
               perspective_embedding.cpp)

target_link_libraries(runCoreTests gtest gtest_main RAS)
