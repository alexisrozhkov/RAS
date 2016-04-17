include_directories(${gtest_SOURCE_DIR}/include
                    ${gtest_SOURCE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR})

add_executable(runCoreTests
               utils/mat_equal_test.cpp
               balls_and_bins.cpp
               perspective_embedding.cpp
               find_polynomials.cpp
               robust_algebraic_segmentation.cpp)

target_link_libraries(runCoreTests
                      gtest
                      gtest_main
                      RAS)
