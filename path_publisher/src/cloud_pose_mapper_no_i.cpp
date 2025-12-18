// ROS Melodic: fuse PointCloud2 + Pose(Odometry/PoseStamped) into a global map
// Build deps: roscpp, sensor_msgs, nav_msgs, geometry_msgs, std_srvs,
//             message_filters, pcl_ros, pcl_conversions, Eigen3

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_srvs/Trigger.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

class CloudPoseMapper {
public:
  CloudPoseMapper(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh)
  {
    // ---- Params (with sane defaults)
    pnh_.param<std::string>("cloud_topic", cloud_topic_, "/velodyne_points");
    pnh_.param<std::string>("pose_topic",  pose_topic_,  "/pose");
    pnh_.param<std::string>("pose_type",   pose_type_,   "Odometry"); // "Odometry" or "PoseStamped"
    pnh_.param<std::string>("map_frame",   map_frame_,   "map");
    pnh_.param<double>("voxel_leaf", leaf_, 0.05);            // meters
    pnh_.param<double>("sync_slop",  sync_slop_, 0.05);       // seconds
    pnh_.param<int>("publish_every_n", publish_every_, 5);    // downsample/publish cadence
    pnh_.param<int>("max_points", max_points_, 2000000);      // trigger downsample when exceeding
    // static extrinsic base_link->lidar (if your pose is base_link and cloud is lidar frame)
    std::vector<double> extr_xyz{0,0,0}, extr_rpy{0,0,0};
    pnh_.getParam("extrinsic_xyz", extr_xyz);
    pnh_.getParam("extrinsic_rpy", extr_rpy);

    T_base_lidar_ = rpyxyzToAffine(extr_rpy, extr_xyz);

    acc_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    acc_->reserve(200000);

    pub_map_ = nh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 1, /*latch=*/true);
    srv_save_ = nh_.advertiseService("save_map", &CloudPoseMapper::onSaveMap, this);

    // ---- Subscribers (+ ApproximateTime sync)
    sub_cloud_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, cloud_topic_, 10));

    if (pose_type_ == "Odometry" || pose_type_ == "odometry") {
      sub_odom_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh_, pose_topic_, 50));
      using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
      sync_odom_.reset(new message_filters::Synchronizer<Policy>(Policy(100), *sub_cloud_, *sub_odom_));
      sync_odom_->setMaxIntervalDuration(ros::Duration(sync_slop_));
      sync_odom_->registerCallback(boost::bind(&CloudPoseMapper::cbCloudOdom, this, _1, _2));
      ROS_INFO_STREAM("mapper: cloud " << cloud_topic_ << " | pose (Odometry) " << pose_topic_);
    } else {
      sub_pose_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh_, pose_topic_, 50));
      using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PoseStamped>;
      sync_pose_.reset(new message_filters::Synchronizer<Policy>(Policy(100), *sub_cloud_, *sub_pose_));
      sync_pose_->setMaxIntervalDuration(ros::Duration(sync_slop_));
      sync_pose_->registerCallback(boost::bind(&CloudPoseMapper::cbCloudPose, this, _1, _2));
      ROS_INFO_STREAM("mapper: cloud " << cloud_topic_ << " | pose (PoseStamped) " << pose_topic_);
    }

    ROS_INFO("mapper: voxel=%.3fm, slop=%.3fs, publish_every=%d, max_points=%d",
             leaf_, sync_slop_, publish_every_, max_points_);
  }

private:
  // --- Utils
  static Eigen::Affine3f rpyxyzToAffine(const std::vector<double>& rpy,
                                        const std::vector<double>& xyz)
  {
    double r = rpy.size() > 0 ? rpy[0] : 0.0;
    double p = rpy.size() > 1 ? rpy[1] : 0.0;
    double y = rpy.size() > 2 ? rpy[2] : 0.0;
    Eigen::AngleAxisf Rx(static_cast<float>(r), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf Ry(static_cast<float>(p), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf Rz(static_cast<float>(y), Eigen::Vector3f::UnitZ());
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.linear() = (Rz * Ry * Rx).toRotationMatrix();
    if (xyz.size() >= 3) {
      T.translation() = Eigen::Vector3f(static_cast<float>(xyz[0]),
                                        static_cast<float>(xyz[1]),
                                        static_cast<float>(xyz[2]));
    }
    return T;
  }

  static Eigen::Affine3f poseToAffine(const geometry_msgs::Pose& pose)
  {
    Eigen::Quaternionf q(static_cast<float>(pose.orientation.w),
                         static_cast<float>(pose.orientation.x),
                         static_cast<float>(pose.orientation.y),
                         static_cast<float>(pose.orientation.z));
    q.normalize();
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() = Eigen::Vector3f(static_cast<float>(pose.position.x),
                                      static_cast<float>(pose.position.y),
                                      static_cast<float>(pose.position.z));
    return T;
  }

  void accumulateAndPublish(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                            const geometry_msgs::Pose& pose,
                            const ros::Time& stamp)
  {
    // 1) Build map<-base transform from pose
    Eigen::Affine3f T_map_base = poseToAffine(pose);
    // 2) Compose base->lidar extrinsic
    Eigen::Affine3f T_map_lidar = T_map_base * T_base_lidar_;

    // 3) Convert cloud and transform into map frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *in);
    if (in->empty()) return;

    pcl::PointCloud<pcl::PointXYZ> transformed;
    pcl::transformPointCloud(*in, transformed, T_map_lidar.matrix());

    // 4) Accumulate
    (*acc_) += transformed;

    // 5) Guard memory
    if (static_cast<int>(acc_->size()) > max_points_) {
      downsampleInPlace();
    }

    // 6) Publish periodically (latched)
    if (++batch_ >= publish_every_) {
      downsampleInPlace();
      sensor_msgs::PointCloud2 out;
      pcl::toROSMsg(*acc_, out);
      out.header.frame_id = map_frame_;
      out.header.stamp = stamp;           // use pose/cloud stamp
      pub_map_.publish(out);
      batch_ = 0;
      ROS_INFO("mapper: published map with %zu pts", acc_->size());
    }
  }

  void downsampleInPlace()
  {
    if (leaf_ <= 0.0) return;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(acc_);
    float L = static_cast<float>(leaf_);
    vg.setLeafSize(L, L, L);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.filter(*filtered);
    acc_.swap(filtered);
  }

  // --- Callbacks
  void cbCloudOdom(const sensor_msgs::PointCloud2ConstPtr& cloud,
                   const nav_msgs::OdometryConstPtr& odom)
  {
    accumulateAndPublish(cloud, odom->pose.pose, odom->header.stamp);
  }

  void cbCloudPose(const sensor_msgs::PointCloud2ConstPtr& cloud,
                   const geometry_msgs::PoseStampedConstPtr& pose)
  {
    accumulateAndPublish(cloud, pose->pose, pose->header.stamp);
  }

  // --- Service to save PCD
  bool onSaveMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
  {
    std::string save_dir = getenv("HOME") ? std::string(getenv("HOME")) + "/maps" : "./maps";
    pnh_.param<std::string>("save_dir", save_dir, save_dir);
    std::string save_name("map");
    pnh_.param<std::string>("save_name", save_name, save_name);

    if (!acc_ || acc_->empty()) {
      res.success = false;
      res.message = "Map is empty.";
      return true;
    }

    // ensure directory exists
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", save_dir.c_str());
    ::system(cmd);

    const std::string path = save_dir + "/" + save_name + ".pcd";
    int ret = pcl::io::savePCDFileBinary(path, *acc_);
    if (ret == 0) {
      res.success = true;
      res.message = "Saved: " + path;
      ROS_INFO("%s", res.message.c_str());
    } else {
      res.success = false;
      res.message = "Failed to save: " + path;
      ROS_ERROR("%s", res.message.c_str());
    }
    return true;
  }

private:
  ros::NodeHandle nh_, pnh_;
  std::string cloud_topic_, pose_topic_, pose_type_, map_frame_;
  double leaf_, sync_slop_;
  int publish_every_, max_points_;
  int batch_ = 0;

  Eigen::Affine3f T_base_lidar_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr acc_;
  ros::Publisher pub_map_;
  ros::ServiceServer srv_save_;

  // Subscribers + sync (only one pair will be used)
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> sub_cloud_;
  std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> sub_odom_;
  std::shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> sub_pose_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>>> sync_odom_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PoseStamped>>> sync_pose_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cloud_pose_mapper");
  ros::NodeHandle nh, pnh("~");
  CloudPoseMapper node(nh, pnh);
  ros::spin();
  return 0;
}