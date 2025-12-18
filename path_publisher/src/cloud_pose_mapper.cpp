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

#include <cmath> 

#include <boost/bind.hpp>


class CloudPoseMapper {
public:
  CloudPoseMapper(ros::NodeHandle& nh, ros::NodeHandle& pnh): nh_(nh), pnh_(pnh) {
    pnh_.param<std::string>("cloud_topic", cloud_topic_, "/velodyne_points");
    pnh_.param<std::string>("pose_topic",  pose_topic_,  "/pose");
    pnh_.param<std::string>("pose_type",   pose_type_,   "Odometry");
    pnh_.param<std::string>("map_frame",   map_frame_,   "map");
    pnh_.param<double>("voxel_leaf", leaf_, 0.05);
    pnh_.param<double>("sync_slop",  sync_slop_, 0.05);
    pnh_.param<int>("publish_every_n", publish_every_, 5);
    pnh_.param<int>("max_points", max_points_, 2000000);

    std::vector<double> extr_xyz{0,0,0}, extr_rpy{0,0,0};
    pnh_.getParam("extrinsic_xyz", extr_xyz);
    pnh_.getParam("extrinsic_rpy", extr_rpy);
    T_base_lidar_ = rpyxyzToAffine(extr_rpy, extr_xyz);

    pnh_.param<bool>("enable_yaw_filter", enable_yaw_filter_, false);
    pnh_.param<double>("yaw_min", yaw_min_, -3.0);
    pnh_.param<double>("yaw_max", yaw_max_,  3.0);
    pnh_.param<bool>("yaw_invert", yaw_invert_, false);
    
    pnh_.param<bool>("enable_downsample", enable_downsample_, true);

    pnh_.param("z_min", z_min_map_, -1e9);
    
    acc_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    acc_->points.reserve(200000);   // instead of acc_->reserve(...)

    pub_map_ = nh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 1, /*latch=*/true);
    srv_save_ = nh_.advertiseService("save_map", &CloudPoseMapper::onSaveMap, this);

    sub_cloud_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, cloud_topic_, 10));

    if (pose_type_ == "Odometry" || pose_type_ == "odometry") {
      sub_odom_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh_, pose_topic_, 50));
      using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
      sync_odom_.reset(new message_filters::Synchronizer<Policy>(Policy(100), *sub_cloud_, *sub_odom_));
      sync_odom_->setMaxIntervalDuration(ros::Duration(sync_slop_));
      sync_odom_->registerCallback(boost::bind(&CloudPoseMapper::cbCloudOdom, this, _1, _2));
    } else {
      sub_pose_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh_, pose_topic_, 50));
      using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PoseStamped>;
      sync_pose_.reset(new message_filters::Synchronizer<Policy>(Policy(100), *sub_cloud_, *sub_pose_));
      sync_pose_->setMaxIntervalDuration(ros::Duration(sync_slop_));
      sync_pose_->registerCallback(boost::bind(&CloudPoseMapper::cbCloudPose, this, _1, _2));
    }

    ROS_INFO("mapper: voxel=%.3fm, slop=%.3fs, publish_every=%d", leaf_, sync_slop_, publish_every_);
  }

private:

  bool   enable_yaw_filter_ = false;
  double yaw_min_ = -3.0;
  double yaw_max_ =  3.0;
  bool   yaw_invert_ = false;

  bool enable_downsample_ = true;

  double z_min_map_ = -1e9;

  static Eigen::Affine3f rpyxyzToAffine(const std::vector<double>& rpy,
                                        const std::vector<double>& xyz) {
    double r = rpy.size()>0 ? rpy[0] : 0.0;
    double p = rpy.size()>1 ? rpy[1] : 0.0;
    double y = rpy.size()>2 ? rpy[2] : 0.0;
    Eigen::AngleAxisf Rx((float)r, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf Ry((float)p, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf Rz((float)y, Eigen::Vector3f::UnitZ());
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.linear() = (Rz * Ry * Rx).toRotationMatrix();
    if (xyz.size()>=3) T.translation() = Eigen::Vector3f((float)xyz[0],(float)xyz[1],(float)xyz[2]);
    return T;
  }

  static Eigen::Affine3f poseToAffine(const geometry_msgs::Pose& pose) {
    Eigen::Quaternionf q((float)pose.orientation.w,
                         (float)pose.orientation.x,
                         (float)pose.orientation.y,
                         (float)pose.orientation.z);
    q.normalize();
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() = Eigen::Vector3f((float)pose.position.x,
                                      (float)pose.position.y,
                                      (float)pose.position.z);
    return T;
  }

  inline static double wrapPi(double a){
    while(a <= -M_PI) a += 2*M_PI;
    while(a >   M_PI) a -= 2*M_PI;
    return a;
  }
  
  inline bool yawInRange(double a, double mn, double mx) const {
    a  = wrapPi(a); mn = wrapPi(mn); mx = wrapPi(mx);
    if (mn <= mx) return (a >= mn && a <= mx);
    // handles wrap-around selections, e.g. mn=2.8, mx=-2.8
    return (a >= mn || a <= mx);
  }

  void cbCloudOdom(const sensor_msgs::PointCloud2ConstPtr& cloud,
                   const nav_msgs::OdometryConstPtr& odom) {
    accumulate(cloud, odom->pose.pose, odom->header.stamp);
  }

  void cbCloudPose(const sensor_msgs::PointCloud2ConstPtr& cloud,
                   const geometry_msgs::PoseStampedConstPtr& pose) {
    accumulate(cloud, pose->pose, pose->header.stamp);
  }
  
  void accumulate(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                  const geometry_msgs::Pose& pose,
                  const ros::Time& stamp)
  {
    // 1) map<-base and map<-lidar
    Eigen::Affine3f T_map_base  = poseToAffine(pose);
    Eigen::Affine3f T_map_lidar = T_map_base * T_base_lidar_;

    // 2) Convert to PCL and ensure intensity is present
    pcl::PCLPointCloud2::Ptr in2(new pcl::PCLPointCloud2);
    pcl_conversions::toPCL(*cloud_msg, *in2);

    bool has_intensity = false;
    for (const auto& f : in2->fields)
      if (f.name == "intensity") { has_intensity = true; break; }

    pcl::PointCloud<pcl::PointXYZI>::Ptr in(new pcl::PointCloud<pcl::PointXYZI>);
    if (has_intensity) {
      pcl::fromPCLPointCloud2(*in2, *in);
    } else {
      pcl::PointCloud<pcl::PointXYZ> in_xyz;
      pcl::fromPCLPointCloud2(*in2, in_xyz);
      in->resize(in_xyz.size());
      for (size_t i=0;i<in_xyz.size();++i) {
        (*in)[i].x = in_xyz[i].x;
        (*in)[i].y = in_xyz[i].y;
        (*in)[i].z = in_xyz[i].z;
        (*in)[i].intensity = 0.f;
      }
    }
    if (in->empty()) return;

    // 3) (NEW) Yaw filter in *sensor frame* AFTER conversion
    if (enable_yaw_filter_) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr kept(new pcl::PointCloud<pcl::PointXYZI>);
      kept->reserve(in->size());
      for (const auto& pt : in->points) {
        double yaw = std::atan2(pt.y, pt.x);
        bool keep = yawInRange(yaw, yaw_min_, yaw_max_);
        if (yaw_invert_) keep = !keep;  // exclude range if requested
        if (keep) kept->push_back(pt);
      }
      in.swap(kept);
      if (in->empty()) return;
    }

    // 4) Transform to map frame and accumulate
    pcl::PointCloud<pcl::PointXYZI> transformed;
    pcl::transformPointCloud(*in, transformed, T_map_lidar.matrix());

    if (z_min_map_ > -1e8) {               // only if user set something sensible
      pcl::PointCloud<pcl::PointXYZI> kept;
      kept.points.reserve(transformed.size());
      for (const auto& pt : transformed.points) {
        if (pt.z >= z_min_map_) kept.push_back(pt);
      }
      transformed.swap(kept);
      if (transformed.empty()) return;
    }

    (*acc_) += transformed;

    // 5) Downsample/publish (guarded)
    if (enable_downsample_ && static_cast<int>(acc_->size()) > max_points_) {
      downsampleInPlace();
    }
    if (++batch_ >= publish_every_) {
      if (enable_downsample_) {
        downsampleInPlace();
      }
      sensor_msgs::PointCloud2 out;
      pcl::toROSMsg(*acc_, out);
      out.header.frame_id = map_frame_;
      out.header.stamp    = stamp;
      pub_map_.publish(out);
      batch_ = 0;
      ROS_INFO("mapper: published map with %zu pts", acc_->size());
    }
  
  }

  void downsampleInPlace() {
    if (!enable_downsample_ || leaf_ <= 0.0 || !acc_ || acc_->empty()) return;
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(acc_);
    float L = static_cast<float>(leaf_);
    vg.setLeafSize(L, L, L);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
    vg.filter(*filtered);
    acc_.swap(filtered);
  }

  bool onSaveMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    std::string save_dir = getenv("HOME") ? std::string(getenv("HOME")) + "/maps" : "./maps";
    pnh_.param<std::string>("save_dir", save_dir, save_dir);
    std::string save_name("map");
    pnh_.param<std::string>("save_name", save_name, save_name);

    if (!acc_ || acc_->empty()) { res.success=false; res.message="Map is empty."; return true; }

    char cmd[512]; snprintf(cmd,sizeof(cmd),"mkdir -p '%s'", save_dir.c_str()); ::system(cmd);
    std::string path = save_dir + "/" + save_name + ".pcd";
    int ret = pcl::io::savePCDFileBinary(path, *acc_);
    if (ret==0) { res.success=true; res.message="Saved: "+path; }
    else        { res.success=false; res.message="Failed to save: "+path; }
    return true;
  }

  ros::NodeHandle nh_, pnh_;
  std::string cloud_topic_, pose_topic_, pose_type_, map_frame_;
  double leaf_, sync_slop_;
  int publish_every_, max_points_, batch_ = 0;

  Eigen::Affine3f T_base_lidar_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr acc_;
  ros::Publisher pub_map_;
  ros::ServiceServer srv_save_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> sub_cloud_;
  std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> sub_odom_;
  std::shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> sub_pose_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>>> sync_odom_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PoseStamped>>> sync_pose_;
};

int main(int argc, char** argv){
  ros::init(argc, argv, "cloud_pose_mapper");
  ros::NodeHandle nh, pnh("~");
  CloudPoseMapper node(nh, pnh);
  ros::spin();
  return 0;
}