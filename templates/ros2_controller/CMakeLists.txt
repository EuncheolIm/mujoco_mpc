cmake_minimum_required(VERSION 3.16)
project(example_mjpc_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

add_library(${PROJECT_NAME} SHARED src/example_mjpc_controller.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)
ament_target_dependencies(${PROJECT_NAME}
  controller_interface hardware_interface pluginlib rclcpp rclcpp_lifecycle)

# shm_open / shm_unlink live in librt on glibc < 2.34. Harmless to link either way,
# and omitting it is an undefined-reference at link time on older distros.
target_link_libraries(${PROJECT_NAME} rt)

# This line is what makes `ros2 control load_controller` able to find the class.
# Without it the plugin builds, installs, and is then simply not found -- the error
# says the controller type does not exist, which does not point here.
pluginlib_export_plugin_description_file(controller_interface
  example_mjpc_controller_plugin.xml)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
