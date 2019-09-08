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
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system filesystem)
add_definitions(-DDEBUG_YES)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INLCUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${fast_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${vikit_common_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)

list(APPEND LIBS
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${fast_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Pangolin_LIBRARIES}
  glog
)

list(APPEND SRCS
  src/depth_filter.cpp
  src/io.cpp
  src/pinhole.cpp
  src/frame.cpp
  src/utils.cpp
  src/viewer.cpp
)
add_library(depth_filter SHARED ${SRCS} src/frame.cpp include/depth_filter/utils.h)
target_link_libraries(depth_filter ${LIBS})

add_executable(test_pipeline test/test_pipeline.cpp)
target_link_libraries(test_pipeline depth_filter)

add_executable(analyze_kitti analysis/kitti.cpp)
target_link_libraries(analyze_kitti depth_filter)

add_executable(viz_cloud analysis/viz_point_cloud.cpp)
target_link_libraries(viz_cloud depth_filter)

add_executable(test_analysis test/test_analysis.cpp)
target_link_libraries(test_analysis depth_filter)

add_executable(test_utils test/test_utils.cpp)
target_link_libraries(test_utils depth_filter)

add_executable(test_fast test/test_fast.cpp)
target_link_libraries(test_fast depth_filter)

add_executable(test_similarity test/test_similarity.cpp)
target_link_libraries(test_similarity depth_filter)

add_executable(test_velo test/test_velodyne_scan_kitti.cpp)
target_link_libraries(test_velo depth_filter)

add_executable(test_df_kitti analysis/depth_filter_kitti.cpp)
target_link_libraries(test_df_kitti depth_filter)

add_executable(depth_filter_synthetic_downward analysis/depth_filter_synthetic_downward.cpp)
target_link_libraries(depth_filter_synthetic_downward depth_filter)

add_executable(depth_filter_synthetic_forward analysis/depth_filter_synthetic_forward.cpp)
target_link_libraries(depth_filter_synthetic_forward depth_filter)

add_executable(search_similarity_downward analysis/search_similarity_downward.cpp)
target_link_libraries(search_similarity_downward depth_filter)

add_executable(search_similarity_forward analysis/search_similarity_forward.cpp)
target_link_libraries(search_similarity_forward depth_filter)
