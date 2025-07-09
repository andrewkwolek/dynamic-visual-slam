cmake_minimum_required(VERSION 3.8)
project(dynamic_visual_slam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(message_filters REQUIRED)
find_package(Ceres REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(dynamic_visual_slam_interfaces REQUIRED)

set(DBoW2_ROOT_DIR "$ENV{HOME}/ws/DBoW2")
find_path(DBoW2_INCLUDE_DIR NAMES DBoW2/DBoW2.h PATHS ${DBoW2_ROOT_DIR}/include)
find_library(DBoW2_LIBRARY NAMES DBoW2 PATHS ${DBoW2_ROOT_DIR}/build NO_DEFAULT_PATH)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${DBoW2_INCLUDE_DIR}
)

add_executable(frontend src/frontend.cpp)
ament_target_dependencies(frontend 
  rclcpp 
  sensor_msgs 
  OpenCV 
  cv_bridge 
  tf2_ros 
  tf2 
  tf2_eigen 
  Eigen3 
  message_filters 
  Ceres
  dynamic_visual_slam_interfaces
)

add_executable(backend src/backend.cpp)
ament_target_dependencies(backend 
  rclcpp
  sensor_msgs
  OpenCV 
  tf2_ros 
  tf2 
  tf2_eigen 
  Eigen3 
  Ceres
  visualization_msgs
  dynamic_visual_slam_interfaces
)

target_link_libraries(backend
  ${CERES_LIBRARIES}
  ${DBoW2_LIBRARIES}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files.
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install nodes
install(
  TARGETS frontend backend
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include/
)

ament_export_include_directories(include)
ament_package()