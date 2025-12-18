#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

class PathPublisher {
public:
  PathPublisher() {
    ros::NodeHandle nh;
    path_pub_ = nh.advertise<nav_msgs::Path>("lego_loam_path", 10);
    odom_sub_ = nh.subscribe("aft_mapped_to_init", 1000, &PathPublisher::odomCallback, this);
    path_.header.frame_id = "map";  // or "camera_init" depending on your RViz config
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = msg->header;
    pose_stamped.pose = msg->pose.pose;
    path_.header.stamp = ros::Time::now();
    path_.poses.push_back(pose_stamped);
    path_pub_.publish(path_);
  }

private:
  ros::Publisher path_pub_;
  ros::Subscriber odom_sub_;
  nav_msgs::Path path_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "path_publisher_node");
  PathPublisher pp;
  ros::spin();
  return 0;
}

