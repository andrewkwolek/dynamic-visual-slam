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
find_package(ament_index_cpp REQUIRED)

# DBoW2 and DLib configuration with known paths
set(DBoW2_ROOT_DIR "$ENV{HOME}/ws/DBoW2")
set(DLib_ROOT_DIR "$ENV{HOME}/ws/DLib")

# Set the exact paths we know exist
set(DLib_INCLUDE_DIR "${DLib_ROOT_DIR}/include")
set(DLib_LIBRARY "${DLib_ROOT_DIR}/build/libDLib.so")
set(DBoW2_INCLUDE_DIR "${DBoW2_ROOT_DIR}/include")
set(DBoW2_LIBRARY "${DBoW2_ROOT_DIR}/build/libDBoW2.so")

# Verify paths exist
if(NOT EXISTS ${DLib_INCLUDE_DIR})
  message(FATAL_ERROR "DLib include directory not found: ${DLib_INCLUDE_DIR}")
endif()

if(NOT EXISTS ${DLib_LIBRARY})
  message(FATAL_ERROR "DLib library not found: ${DLib_LIBRARY}")
endif()

if(NOT EXISTS ${DBoW2_INCLUDE_DIR})
  message(FATAL_ERROR "DBoW2 include directory not found: ${DBoW2_INCLUDE_DIR}")
endif()

if(NOT EXISTS ${DBoW2_LIBRARY})
  message(FATAL_ERROR "DBoW2 library not found: ${DBoW2_LIBRARY}")
endif()

# Debug output
message(STATUS "Found DLib include: ${DLib_INCLUDE_DIR}")
message(STATUS "Found DLib library: ${DLib_LIBRARY}")
message(STATUS "Found DBoW2 include: ${DBoW2_INCLUDE_DIR}")
message(STATUS "Found DBoW2 library: ${DBoW2_LIBRARY}")

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${DBoW2_INCLUDE_DIR}
  ${DLib_INCLUDE_DIR}
)

add_library(orb_slam3_components
  src/ORBextractor.cpp
)

target_include_directories(orb_slam3_components PUBLIC
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(orb_slam3_components
  ${OpenCV_LIBRARIES}
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

target_link_libraries(frontend
  orb_slam3_components
)

# Link libraries to backend
target_link_libraries(backend
  ${CERES_LIBRARIES}
  ${DBoW2_LIBRARY}
  ${DLib_LIBRARY}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  
  # Add the test executable
  ament_add_gtest(test_dbow2_integration test/test_dbow2_integration.cpp)
  
  target_include_directories(test_dbow2_integration PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${DBoW2_INCLUDE_DIR}
    ${DLib_INCLUDE_DIR}
  )
  
  target_link_libraries(test_dbow2_integration
    ${OpenCV_LIBRARIES}
    ${DBoW2_LIBRARY}
    ${DLib_LIBRARY}
  )
  
  ament_target_dependencies(test_dbow2_integration
    rclcpp
    OpenCV
    ament_index_cpp
  )
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