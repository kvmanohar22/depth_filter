project(depth_estimation)
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fast REQUIRED)
find_package(Sophus REQUIRED)
find_package(vikit_common REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system)
add_definitions(-DDEBUG_YES)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INLCUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${fast_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${vikit_common_INCLUDE_DIRS}
)

list(APPEND LIBS
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${fast_LIBRARIES}
  ${Sophus_LIBRARIES}
)

list(APPEND SRCS
  src/depth_filter.cpp
  src/io.cpp
  src/pinhole.cpp
  src/frame.cpp
)
add_library(depth_filter SHARED ${SRCS} src/frame.cpp include/depth_filter/utils.h)
target_link_libraries(depth_filter ${LIBS})

add_executable(test_pipeline test/test_pipeline.cpp)
target_link_libraries(test_pipeline depth_filter)

add_executable(test_analysis test/test_analysis.cpp)
target_link_libraries(test_analysis depth_filter)

add_executable(test_utils test/test_utils.cpp)
target_link_libraries(test_utils depth_filter)

