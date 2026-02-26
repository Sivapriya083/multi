cmake_minimum_required(VERSION 3.8)
project(ur5e_impulse_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ament_cmake: core ROS 2 CMake macros
# ament_cmake_python NOT needed - we have no shared Python module,
# just 3 standalone script files installed directly as executables.
find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/controller_node.py
  scripts/injector_node.py
  scripts/monitor_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY worlds/  DESTINATION share/${PROJECT_NAME}/worlds)


ament_package()