cmake_minimum_required(VERSION 2.6)
project(point_cloud_2)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(CSparse REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(G2O REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
  "/usr/include/eigen3/"
)

add_definitions( ${PCL_DEFINITIONS})

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")#解决pcl找不到 -lvtkproj4

add_executable(point_cloud_2 main.cpp)
target_link_libraries(point_cloud_2
  ${OpenCV_LIBS}
  ${CSPARSE_LIBRARY}
  ${PCL_LIBRARIES}
  g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)
  