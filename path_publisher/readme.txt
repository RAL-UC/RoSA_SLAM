here I have two packages:
1.



2. Publish 3D map from poses and velodyne point cloud
    roscore
    On the older where the rosbags are recorded: rosrun rqt_bag rqt_bag
    add the rosbags (C1) C1_2024-10-17-12-45-05.bag ; poses.bag
    and publish the topics: /velodyne_point_cloud and poses
    roslaunch path_publisher cloud_pose_mapper.launch
    rosservice call /cloud_pose_mapper/save_map (I didn't use it.. but its suppose that save the point cloud)
    rviz  add - map point cloud (we can visualize with intensity or with Color Transformer: AxisColor; Axis: Z; Adjust Min/Max Value to your map’s Z range)