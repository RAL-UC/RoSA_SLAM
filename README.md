# RoSA_SLAM
This repository contains a Python-based Robust SLAM for Arboreal environments, combining 3D point cloud slicing and 2D Hausdorff scan-to-map matching for reliable navigation using only LiDAR data. The system consumes point clouds from a horizontally oriented Velodyne VLP-16 and produces accurate pose estimations, along with a 3D map, without relying on IMU or GNSS measurements.

<p align="center">
  <img src="pictures/Robust-Navigation-in-Arboreal-Environments.gif" width="70%">
</p>

## 1. Dependency

The algorithm was tested with:

Operating System:   Ubuntu 18.04 LTS  
Architecture:       x86_64
ROS Distribution:   ROS Melodic

System & ROS Dependencies
| Component     | Version / Notes     |
| ------------- | ------------------- |
| ROS           | Melodic             |
| roscpp        | ROS Melodic default |
| rospy         | ROS Melodic default |
| tf / tf2      | ROS Melodic default |
| sensor_msgs   | ROS Melodic default |
| geometry_msgs | ROS Melodic default |

Python Dependencies
| Package         | Version        |
|-----------------|----------------|
| Python          | 3.8.3          |
| fonttools       | 4.44.0         |
| ipython         | 8.12.3         |
| jupyter_client  | 8.6.0          |
| jupyter_core    | 5.5.0          |
| matplotlib      | 3.7.3          |
| matplotlib-inline | 0.1.6        |
| numba           | 0.58.1         |
| numpy           | 1.24.4         |
| open3d          | 0.18.0         |
| opencv-python   | 4.8.1.78       |
| rosbag          | 1.14.13        |
| rospy           | 1.14.13        |
| scikit-learn    | 1.3.2          |
| scipy           | 1.10.1         |


## 2. Build

Prerequisites:
- Ubuntu 18.04 LTS
- ROS Melodic (desktop-full)

Clone the repository and catkin_make:
```bash
git clone https://github.com/RAL-UC/RoSA_SLAM.git
cd path/RoSA_SLAM/path_publisher
cp -r path_publisher ~/catkin_ws/src
cd ~/catkin_ws/src
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. Project Structure

```text
├── data/                        # Input dataset
├── path_publisher/              # ROS package
│   ├── launch/                  # Launch files
│   ├── rviz/                    # RViz configuration
│   └── src/                     # Source code
├── pictures/                    # imagens of readme
├── pullally_example/            # Python code
│   ├── EKF/                     # EKF and robot model functions
│   ├── hausdorff/               # Trajectory matching
│   ├── utils/                   # Screen and variable utilities
│   └── example_pullally.ipynb   # Main example with output files and figures
├── results_evaluation.ipynb     # Output files and figures of some datasets
└── README.md
```

## 4. Example

To run the Python code, use:
```text
example_pullally.ipynb
```

Download the dataset from Link_to_pullally_dataset and store it in YOUR_DATASET_FOLDER.
```bash
roscore
roslaunch path_publisher cloud_pose_mapper.launch
rosbag play data_pullally_example.bag
```

<p align="center">
  <img src="pictures/map_ros.gif" width="70%">
</p>

    
## 5. Cite
Nazate-Burgos, P., Torres-Torriti, M., Aguilera-Marinovic, S., Arévalo, T., Huang, S., & Auat Cheein, F. (2025). Robust 2D lidar-based SLAM in arboreal environments without IMU/GNSS. arXiv. arXiv:2505.10847.
