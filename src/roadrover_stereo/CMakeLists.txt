cmake_minimum_required(VERSION 3.8)
project(roadrover_stereo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp      REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs    REQUIRED)
find_package(OpenCV      REQUIRED)

add_executable(stereo_splitter src/stereo_splitter.cpp)

target_include_directories(stereo_splitter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(stereo_splitter PUBLIC cxx_std_17)

ament_target_dependencies(stereo_splitter
  rclcpp
  sensor_msgs
  std_msgs
  OpenCV)

install(TARGETS stereo_splitter
  DESTINATION lib/${PROJECT_NAME})

ament_package()
