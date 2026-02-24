cmake_minimum_required(VERSION 3.8)
project(ur5e_dual_arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Find required packages ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# =============================================================================
# CUSTOM CONTROLLER SHARED LIBRARY (.so plugin)
# =============================================================================

add_library(dual_arm_impedance_controller SHARED
  src/dual_arm_impedance_controller.cpp
)

target_include_directories(dual_arm_impedance_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(dual_arm_impedance_controller
  rclcpp
  controller_interface
  hardware_interface
  pluginlib
  std_msgs
  sensor_msgs
  geometry_msgs
)

# Register plugin XML so controller_manager can discover this controller
pluginlib_export_plugin_description_file(
  controller_interface
  dual_arm_plugin.xml
)

# =============================================================================
# INSTALL RULES
# =============================================================================

# Compiled controller .so library
install(TARGETS dual_arm_impedance_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Plugin XML
install(FILES dual_arm_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

# C++ headers
install(DIRECTORY include/
  DESTINATION include
)

# Python scripts — installed directly as executable programs (no setup.py needed)
install(PROGRAMS
  src/scripts/task_executor.py
  src/scripts/data_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch, config, urdf, worlds directories
install(DIRECTORY
  launch
  config
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

# =============================================================================
# EXPORT
# =============================================================================

ament_export_include_directories(include)
ament_export_libraries(dual_arm_impedance_controller)
ament_export_dependencies(
  controller_interface
  hardware_interface
  pluginlib
  std_msgs
  sensor_msgs
)

ament_package()