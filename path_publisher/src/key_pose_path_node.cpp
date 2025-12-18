#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class KeyPosePathPublisher {
public:
  KeyPosePathPublisher() {
    ros::NodeHandle nh;
    path_pub_ = nh.advertise<nav_msgs::Path>("key_pose_path", 10);
    sub_ = nh.subscribe("key_pose_origin", 1, &KeyPosePathPublisher::callback, this);
    path_.header.frame_id = "map";  // or "camera_init"
  }

  void callback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    for (const auto& pt : pcl_cloud.points) {
      if (!isNewPoint(pt)) continue;

      geometry_msgs::PoseStamped pose;
      pose.header.stamp = ros::Time::now();
      pose.header.frame_id = path_.header.frame_id;
      //pose.pose.position.x = pt.x;  //pt.y;
      //pose.pose.position.y = pt.y;  //pt.z;
      //pose.pose.position.z = pt.z;  //pt.x;
      pose.pose.position.x = pt.z;
      pose.pose.position.y = pt.x;
      pose.pose.position.z = pt.y;
      pose.pose.orientation.w = 1.0;
      path_.poses.push_back(pose);
      previous_points_.push_back(pt);
    }

    path_.header.stamp = ros::Time::now();
    path_pub_.publish(path_);
  }

private:
  bool isNewPoint(const pcl::PointXYZI& pt) {
    for (const auto& prev : previous_points_) {
      if (fabs(prev.x - pt.x) < 1e-3 &&
          fabs(prev.y - pt.y) < 1e-3 &&
          fabs(prev.z - pt.z) < 1e-3) {
        return false;
      }
    }
    return true;
  }

  ros::Publisher path_pub_;
  ros::Subscriber sub_;
  nav_msgs::Path path_;
  std::vector<pcl::PointXYZI> previous_points_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "key_pose_path_node");
  KeyPosePathPublisher node;
  ros::spin();
  return 0;
}