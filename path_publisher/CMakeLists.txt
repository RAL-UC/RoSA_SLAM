cmake_minimum_required(VERSION 2.8.3)
project(path_publisher)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  std_srvs
  message_filters
)


find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_executable(path_publisher_node src/path_publisher_node.cpp)
target_link_libraries(path_publisher_node ${catkin_LIBRARIES})

add_executable(key_pose_path_node src/key_pose_path_node.cpp)
target_link_libraries(key_pose_path_node ${catkin_LIBRARIES})

add_executable(cloud_pose_mapper src/cloud_pose_mapper.cpp)
target_link_libraries(cloud_pose_mapper ${catkin_LIBRARIES})
