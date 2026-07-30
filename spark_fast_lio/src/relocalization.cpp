#include "relocalization.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/msg/bool.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace spark_fast_lio {

namespace {

Eigen::Isometry3d poseMsgToIso(const geometry_msgs::msg::Pose& p) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation()     = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
  Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  if (q.norm() < 1e-9) q = Eigen::Quaterniond::Identity();
  q.normalize();
  T.linear() = q.toRotationMatrix();
  return T;
}

geometry_msgs::msg::TransformStamped isoToTfMsg(const Eigen::Isometry3d& T,
                                                const rclcpp::Time& stamp,
                                                const std::string& parent,
                                                const std::string& child) {
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp            = stamp;
  tf.header.frame_id         = parent;
  tf.child_frame_id          = child;
  tf.transform.translation.x = T.translation().x();
  tf.transform.translation.y = T.translation().y();
  tf.transform.translation.z = T.translation().z();
  Eigen::Quaterniond q(T.linear());
  q.normalize();
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  tf.transform.rotation.w = q.w();
  return tf;
}

// Parse a mapping trajectory into map-frame positions. Supports TUM
// (t x y z qx qy qz qw) and KITTI (12-value row-major 3x4 [R|t]); "auto" picks
// KITTI when a line has >=12 numbers, otherwise TUM. Comment lines (#) skipped.
bool parseTrajectory(const std::string& path, const std::string& fmt,
                     std::vector<Eigen::Vector3d>& out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::vector<double> v;
    double x;
    while (ss >> x) v.push_back(x);
    const std::string use_fmt =
        (fmt == "auto") ? (v.size() >= 12 ? "kitti" : "tum") : fmt;
    if (use_fmt == "kitti" && v.size() >= 12) {
      out.emplace_back(v[3], v[7], v[11]);
    } else if (v.size() >= 4) {  // tum: t x y z ...
      out.emplace_back(v[1], v[2], v[3]);
    }
  }
  return !out.empty();
}

// Look for a trajectory next to the prior map .pcd. map-frame files preferred.
std::string autoDetectTrajectory(const std::string& map_file) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(map_file).parent_path();
  for (const char* c : {"poses_tum.txt", "poses_map.txt", "poses_kitti.txt"}) {
    const fs::path p = dir / c;
    if (fs::exists(p)) return p.string();
  }
  return "";
}

}  // namespace

Relocalization::Relocalization(const rclcpp::NodeOptions& options)
    : rclcpp::Node("spark_lio_relocalization", options),
      prior_map_ds_(new PointCloudT),
      latest_scan_in_odom_(new PointCloudT),
      latest_scan_stamp_(0, 0, RCL_ROS_TIME),
      latest_odom_stamp_(0, 0, RCL_ROS_TIME) {
  map_file_   = declare_parameter<std::string>("relocalization.map_file", "");
  map_frame_  = declare_parameter<std::string>("relocalization.map_frame", "map");
  odom_frame_ = declare_parameter<std::string>("relocalization.odom_frame", "odom");
  base_frame_ = declare_parameter<std::string>("relocalization.base_frame", "base_link");

  // Namespace the TF frames like spark_fast_lio.cpp: a relative frame gets the node
  // namespace prepended (map -> unitree2/map); a leading '/' opts out (slash stripped,
  // since tf2 disallows leading slashes in frame ids).
  {
    std::string ns = get_namespace();                       // "/unitree2" or "/"
    ns             = (ns == "/") ? "" : ns.substr(1) + "/"; // "unitree2/" or ""
    auto qualify   = [&](std::string& f) {
      if (f.empty()) return;
      if (f.front() == '/') f = f.substr(1);
      else if (!ns.empty()) f = ns + f;
    };
    qualify(map_frame_);
    qualify(odom_frame_);
    qualify(base_frame_);
  }

  prior_map_voxel_size_ =
      declare_parameter<double>("relocalization.prior_map_voxel_size", 0.4);
  scan_voxel_size_ = declare_parameter<double>("relocalization.scan_voxel_size", 0.4);
  max_corresp_dist_init_ =
      declare_parameter<double>("relocalization.gicp.max_correspondence_distance_init", 5.0);
  max_corresp_dist_track_ =
      declare_parameter<double>("relocalization.gicp.max_correspondence_distance_track", 1.0);
  max_iterations_ = declare_parameter<int>("relocalization.gicp.max_iterations", 50);
  transformation_epsilon_ =
      declare_parameter<double>("relocalization.gicp.transformation_epsilon", 1e-4);
  fitness_threshold_ =
      declare_parameter<double>("relocalization.gicp.fitness_threshold", 1.0);
  correction_rate_hz_ =
      declare_parameter<double>("relocalization.correction_rate_hz", 1.0);
  tf_publish_rate_hz_ =
      declare_parameter<double>("relocalization.tf_publish_rate_hz", 50.0);
  publish_localized_odom_ =
      declare_parameter<bool>("relocalization.publish_localized_odom", true);
  coverage_gate_enabled_ =
      declare_parameter<bool>("relocalization.coverage_gate_enabled", false);
  coverage_radius_ = declare_parameter<double>("relocalization.coverage_radius", 15.0);
  trajectory_file_ = declare_parameter<std::string>("relocalization.trajectory_file", "");
  trajectory_format_ =
      declare_parameter<std::string>("relocalization.trajectory_format", "auto");

  const auto initialpose_topic =
      declare_parameter<std::string>("relocalization.initialpose_topic", "/initialpose");
  const auto cloud_topic =
      declare_parameter<std::string>("topics.cloud_registered", "fast_lio/cloud_registered");
  const auto odom_topic =
      declare_parameter<std::string>("topics.odometry", "fast_lio/odometry");
  const auto prior_map_topic =
      declare_parameter<std::string>("relocalization.prior_map_topic", "/prior_map");
  const auto prior_path_topic =
      declare_parameter<std::string>("relocalization.prior_path_topic", "fast_lio/prior_path");
  const auto localized_odom_topic =
      declare_parameter<std::string>("relocalization.localized_odom_topic",
                                     "/localized_odometry");
  const auto status_topic =
      declare_parameter<std::string>("relocalization.status_topic",
                                     "fast_lio/relocalization_status");

  // Load prior map
  if (map_file_.empty()) {
    RCLCPP_FATAL(get_logger(), "relocalization.map_file is empty");
    throw std::runtime_error("relocalization.map_file is empty");
  }
  PointCloudT::Ptr raw(new PointCloudT);
  if (pcl::io::loadPCDFile<PointT>(map_file_, *raw) < 0) {
    RCLCPP_FATAL(get_logger(), "Failed to load prior map: %s", map_file_.c_str());
    throw std::runtime_error("Failed to load prior map");
  }
  RCLCPP_INFO(get_logger(), "Loaded prior map %s with %zu points",
              map_file_.c_str(), raw->size());
  prior_map_ds_ = voxelDownsample(raw, prior_map_voxel_size_);
  RCLCPP_INFO(get_logger(), "Downsampled prior map to %zu points (leaf=%.3f)",
              prior_map_ds_->size(), prior_map_voxel_size_);

  // Prior trajectory (proxy for where the map has data; also published as a path).
  // Loaded independently of the gate so the prior path is always available.
  {
    const std::string traj = trajectory_file_.empty() ? autoDetectTrajectory(map_file_)
                                                       : trajectory_file_;
    if (!traj.empty()) parseTrajectory(traj, trajectory_format_, traj_positions_);
    if (coverage_gate_enabled_ && traj_positions_.empty()) {
      RCLCPP_WARN(get_logger(),
                  "coverage_gate_enabled but no trajectory found next to the map; gate OFF.");
      coverage_gate_enabled_ = false;
    } else if (!traj_positions_.empty()) {
      RCLCPP_INFO(get_logger(), "Loaded prior trajectory: %zu pts from %s (gate %s).",
                  traj_positions_.size(), traj.c_str(),
                  coverage_gate_enabled_ ? "ON" : "OFF");
    }
  }

  // Publishers
  rclcpp::QoS latched_qos(1);
  latched_qos.transient_local().reliable();
  pub_prior_map_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(prior_map_topic, latched_qos);
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*prior_map_ds_, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp    = now();
    pub_prior_map_->publish(msg);
  }
  // Prior path: same latched, one-shot publish as the prior map. Only the mapping
  // trajectory positions are known, so poses carry identity orientation.
  pub_prior_path_ = create_publisher<nav_msgs::msg::Path>(prior_path_topic, latched_qos);
  if (!traj_positions_.empty()) {
    nav_msgs::msg::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp    = now();
    path.poses.reserve(traj_positions_.size());
    for (const auto& p : traj_positions_) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header             = path.header;
      ps.pose.position.x    = p.x();
      ps.pose.position.y    = p.y();
      ps.pose.position.z    = p.z();
      ps.pose.orientation.w = 1.0;
      path.poses.push_back(ps);
    }
    pub_prior_path_->publish(path);
    RCLCPP_INFO(get_logger(), "Published prior path (%zu poses) on %s.",
                path.poses.size(), prior_path_topic.c_str());
  }
  if (publish_localized_odom_) {
    pub_localized_odom_ =
        create_publisher<nav_msgs::msg::Odometry>(localized_odom_topic, rclcpp::QoS(20));
  }
  pub_status_ = create_publisher<std_msgs::msg::Bool>(status_topic, latched_qos);
  {
    std_msgs::msg::Bool b;
    b.data = false;
    pub_status_->publish(b);
  }

  // Subscriptions
  sub_initialpose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      initialpose_topic, rclcpp::QoS(1),
      std::bind(&Relocalization::initialPoseCallback, this, std::placeholders::_1));
  sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic, rclcpp::SensorDataQoS(),
      std::bind(&Relocalization::cloudCallback, this, std::placeholders::_1));
  sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, rclcpp::QoS(50),
      std::bind(&Relocalization::odomCallback, this, std::placeholders::_1));

  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  correction_thread_ = std::thread(&Relocalization::correctionLoop, this);

  RCLCPP_INFO(get_logger(),
              "Relocalization node ready. Awaiting %s in frame '%s'.",
              initialpose_topic.c_str(), map_frame_.c_str());
}

Relocalization::~Relocalization() {
  stop_thread_.store(true);
  if (correction_thread_.joinable()) correction_thread_.join();
}

Relocalization::PointCloudT::Ptr Relocalization::voxelDownsample(
    const PointCloudT::Ptr& in, double leaf) const {
  if (!in || in->empty()) return in;
  PointCloudT::Ptr clean(new PointCloudT);
  clean->reserve(in->size());
  for (const auto& p : in->points) {
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
      clean->points.push_back(p);
    }
  }
  clean->width    = static_cast<uint32_t>(clean->points.size());
  clean->height   = 1;
  clean->is_dense = true;
  if (leaf <= 0.0 || clean->empty()) return clean;
  PointCloudT::Ptr out(new PointCloudT);
  pcl::VoxelGrid<PointT> vg;
  vg.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
  vg.setInputCloud(clean);
  vg.filter(*out);
  return out;
}

void Relocalization::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  PointCloudT::Ptr cloud(new PointCloudT);
  pcl::fromROSMsg(*msg, *cloud);
  std::lock_guard<std::mutex> lk(mutex_);
  latest_scan_in_odom_ = cloud;
  latest_scan_stamp_   = msg->header.stamp;
  has_scan_            = true;
}

void Relocalization::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_odom_T_body_ = poseMsgToIso(msg->pose.pose);
    latest_odom_stamp_  = msg->header.stamp;
    has_odom_           = true;
  }
  broadcastMapToOdomTf(msg->header.stamp);
  publishLocalizedOdom(msg->header.stamp);
}

void Relocalization::initialPoseCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg) {
  RCLCPP_INFO(get_logger(), "Received initial pose in frame '%s'", msg->header.frame_id.c_str());

  PointCloudT::Ptr scan_copy;
  Eigen::Isometry3d odom_T_body;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!has_scan_ || !has_odom_) {
      RCLCPP_WARN(get_logger(),
                  "No scan/odom yet (scan=%d, odom=%d). Ignoring initial pose.",
                  has_scan_, has_odom_);
      return;
    }
    scan_copy   = latest_scan_in_odom_;
    odom_T_body = latest_odom_T_body_;
  }

  const Eigen::Isometry3d map_T_body_guess = poseMsgToIso(msg->pose.pose);
  const Eigen::Isometry3d map_T_odom_guess = map_T_body_guess * odom_T_body.inverse();

  Eigen::Isometry3d map_T_odom_new;
  double fitness = 0.0;
  if (runGicp(scan_copy, map_T_odom_guess, max_corresp_dist_init_, map_T_odom_new, fitness)) {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      map_T_odom_ = map_T_odom_new;
    }
    relocalized_.store(true);
    if (pub_status_) {
      std_msgs::msg::Bool b;
      b.data = true;
      pub_status_->publish(b);
    }
    RCLCPP_INFO(get_logger(), "Relocalized. Fitness=%.4f", fitness);
  } else {
    RCLCPP_WARN(get_logger(),
                "Initial GICP failed (fitness=%.4f > %.4f). Keeping previous state.",
                fitness, fitness_threshold_);
  }
}

bool Relocalization::runGicp(const PointCloudT::Ptr& scan_in_odom,
                             const Eigen::Isometry3d& map_T_odom_guess,
                             double max_corresp_dist,
                             Eigen::Isometry3d& map_T_odom_out,
                             double& fitness_out) {
  if (!scan_in_odom || scan_in_odom->empty() || !prior_map_ds_ || prior_map_ds_->empty()) {
    return false;
  }
  PointCloudT::Ptr scan_ds = voxelDownsample(scan_in_odom, scan_voxel_size_);

  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
  gicp.setMaxCorrespondenceDistance(max_corresp_dist);
  gicp.setMaximumIterations(max_iterations_);
  gicp.setTransformationEpsilon(transformation_epsilon_);
  gicp.setInputSource(scan_ds);
  gicp.setInputTarget(prior_map_ds_);

  PointCloudT aligned;
  const Eigen::Matrix4f guess = map_T_odom_guess.matrix().cast<float>();
  gicp.align(aligned, guess);

  fitness_out = gicp.getFitnessScore();
  if (!gicp.hasConverged() || !std::isfinite(fitness_out) ||
      fitness_out > fitness_threshold_) {
    return false;
  }
  Eigen::Matrix4d T = gicp.getFinalTransformation().cast<double>();
  map_T_odom_out.matrix() = T;
  return true;
}

void Relocalization::broadcastMapToOdomTf(const rclcpp::Time& stamp) {
  Eigen::Isometry3d map_T_odom;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    map_T_odom = map_T_odom_;
  }
  tf_broadcaster_->sendTransform(isoToTfMsg(map_T_odom, stamp, map_frame_, odom_frame_));
}

void Relocalization::publishLocalizedOdom(const rclcpp::Time& stamp) {
  if (!publish_localized_odom_ || !pub_localized_odom_) return;
  Eigen::Isometry3d map_T_odom;
  Eigen::Isometry3d odom_T_body;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    map_T_odom  = map_T_odom_;
    odom_T_body = latest_odom_T_body_;
  }
  const Eigen::Isometry3d map_T_body = map_T_odom * odom_T_body;
  nav_msgs::msg::Odometry out;
  out.header.stamp         = stamp;
  out.header.frame_id      = map_frame_;
  out.child_frame_id       = base_frame_;
  out.pose.pose.position.x = map_T_body.translation().x();
  out.pose.pose.position.y = map_T_body.translation().y();
  out.pose.pose.position.z = map_T_body.translation().z();
  Eigen::Quaterniond q(map_T_body.linear());
  q.normalize();
  out.pose.pose.orientation.x = q.x();
  out.pose.pose.orientation.y = q.y();
  out.pose.pose.orientation.z = q.z();
  out.pose.pose.orientation.w = q.w();
  pub_localized_odom_->publish(out);
}

bool Relocalization::insideCoverage(const Eigen::Vector3d& p_map) const {
  if (traj_positions_.empty()) return true;
  // ponytail: linear nearest-point scan; swap for a kd-tree if trajectories get huge.
  double best_sq = std::numeric_limits<double>::max();
  for (const auto& t : traj_positions_) {
    best_sq = std::min(best_sq, (t - p_map).squaredNorm());
  }
  return best_sq <= coverage_radius_ * coverage_radius_;
}

void Relocalization::correctionLoop() {
  const double period_s     = 1.0 / std::max(1e-3, correction_rate_hz_);
  const auto period_chrono  = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::duration<double>(period_s));
  while (!stop_thread_.load() && rclcpp::ok()) {
    std::this_thread::sleep_for(period_chrono);
    if (stop_thread_.load()) break;
    if (!relocalized_.load()) continue;

    PointCloudT::Ptr scan_copy;
    Eigen::Isometry3d map_T_odom_guess;
    Eigen::Isometry3d odom_T_body;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (!has_scan_) continue;
      scan_copy        = latest_scan_in_odom_;
      map_T_odom_guess = map_T_odom_;
      odom_T_body      = latest_odom_T_body_;
    }

    // Coverage gate: outside the mapped area, scan-to-prior-map GICP has no valid
    // target and would corrupt the correction. Freeze map<-odom (odom dead-reckons
    // the robot forward) and resume once back inside.
    if (coverage_gate_enabled_) {
      const Eigen::Vector3d p_map = (map_T_odom_guess * odom_T_body).translation();
      const bool inside = insideCoverage(p_map);
      if (inside != inside_coverage_.exchange(inside)) {
        RCLCPP_INFO(get_logger(), inside
            ? "Back inside prior-map coverage; resuming scan-to-map correction."
            : "Left prior-map coverage; pausing scan-to-map correction (odom dead-reckons).");
      }
      if (!inside) continue;
    }

    Eigen::Isometry3d map_T_odom_new;
    double fitness = 0.0;
    if (runGicp(scan_copy, map_T_odom_guess, max_corresp_dist_track_, map_T_odom_new, fitness)) {
      std::lock_guard<std::mutex> lk(mutex_);
      map_T_odom_ = map_T_odom_new;
      RCLCPP_DEBUG(get_logger(), "Tracking correction OK. Fitness=%.4f", fitness);
    } else {
      RCLCPP_DEBUG(get_logger(), "Tracking GICP rejected. Fitness=%.4f", fitness);
    }
  }
}

}  // namespace spark_fast_lio

RCLCPP_COMPONENTS_REGISTER_NODE(spark_fast_lio::Relocalization)
