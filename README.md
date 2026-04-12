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
- ROS Melodic (desktop-full) or melodic-ros-base with pcl_ros and pcl_conversions

Clone the repository and catkin_make:
```bash
git clone https://github.com/RAL-UC/RoSA_SLAM.git
cd RoSA_SLAM/
cp -r path_publisher ~/catkin_ws/src
cd ~/catkin_ws
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

Download the dataset from [Rosbag data_pullally_example.bag](https://uccl0-my.sharepoint.com/:f:/r/personal/mtorreto_uc_cl/Documents/Datasets/Pullally_Dataset/Pullally_dataset/Pullally_20230806/data_example4RoSA_SLAM?csf=1&web=1&e=vAS1UE)  of [Pullally Dataset](https://github.com/RAL-UC/Pullally_Dataset) and store it in YOUR_DATASET_FOLDER.
```bash
roscore
roslaunch path_publisher cloud_pose_mapper.launch
rosbag play data_pullally_example.bag
```

<p align="center">
  <img src="pictures/map_ros.gif" width="70%">
</p>


## 5. Docker

Prerequisites:

- Docker (v20+ recommended)

Dowload the Docker Image from [Docker_RoSA](https://uccl0-my.sharepoint.com/:f:/r/personal/mtorreto_uc_cl/Documents/Datasets/Pullally_Dataset/Pullally_dataset/Pullally_20230806/Docker_image?csf=1&web=1&e=KNR88x), and load it on the destination computer:

Terminal 1:

```bash
docker load < ros-melodic-18_RoSA.tar
```
After loading the Docker image, start the container using the commands below.

- Replace <PATH_ROSBAG_IN_YOUR_PC> with the absolute path to the directory on your machine that contains the rosbag files.
For example: /home/dataset_pullally.

- Execute the following commands:

```bash
xhost +local:docker
docker run -it \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --net=host \
  -v /<PATH_ROSBAG_IN_YOUR_PC>:/data/rosbags:ro \
  ros-melodic:18.04
roscore
```

The rosbag directory will be mounted inside the container at /data/rosbags in read-only mode.


Terminal 2:

Identify the running container:
```bash
docker ps
```

- Replace <YOUR_CONTAINER> with the id in your machine. For example: 0f59b1840653.

- Source the ROS environment and launch the application:

```bash
docker exec -it <YOUR_CONTAINER> /bin/bash
source /opt/ros/melodic/setup.bash
roslaunch path_publisher cloud_pose_mapper.launch
```


Terminal 3:

- Attach to the same container and play the rosbag file:

```bash
docker exec -it <YOUR_CONTAINER> /bin/bash
rosbag play data_pullally_example.bag
```

## 6. Cite
Nazate-Burgos, P., Torres-Torriti, M., Huang, S., & Auat Cheein, F. (2026). **Consistent lidar-only SLAM for legged agricultural robots in arboreal environments via robust dimensionality reduction. *Computers and Electronics in Agriculture*, vol. 247, 111687, ISSN 0168-1699, https://doi.org/10.1016/j.compag.2026.

```bibtex
@article{RoSA_SLAM_2026,
title = {Consistent lidar-only SLAM for legged agricultural robots in arboreal environments via robust dimensionality reduction},
journal = {Computers and Electronics in Agriculture},
volume = {247},
pages = {111687},
year = {2026},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2026.111687},
url = {https://www.sciencedirect.com/science/article/pii/S0168169926002826},
author = {Paola Nazate-Burgos and Miguel Torres-Torriti and Shoudong Huang and Fernando {Auat Cheein}},
}