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
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace spark_fast_lio {

using reloc::CorrectionVerdict;

namespace {

constexpr double kDeg = M_PI / 180.0;

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
    const std::string use_fmt = (fmt == "auto") ? (v.size() >= 12 ? "kitti" : "tum") : fmt;
    if (use_fmt == "kitti" && v.size() >= 12) {
      out.emplace_back(v[3], v[7], v[11]);
    } else if (v.size() >= 4) {  // tum: t x y z ...
      out.emplace_back(v[1], v[2], v[3]);
    }
  }
  return !out.empty();
}

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

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Relocalization::Relocalization(const rclcpp::NodeOptions& options)
    : rclcpp::Node("spark_lio_relocalization", options),
      prior_map_ds_(new PointCloudT),
      latest_odom_stamp_(0, 0, RCL_ROS_TIME) {
  map_file_   = declare_parameter<std::string>("relocalization.map_file", "");
  map_frame_  = declare_parameter<std::string>("relocalization.map_frame", "map");
  odom_frame_ = declare_parameter<std::string>("relocalization.odom_frame", "odom");
  base_frame_ = declare_parameter<std::string>("relocalization.base_frame", "base_link");

  // Namespace the TF frames like spark_fast_lio.cpp: a relative frame gets the
  // node namespace prepended (map -> unitree2/map); a leading '/' opts out
  // (slash stripped, since tf2 disallows leading slashes in frame ids).
  {
    std::string ns = get_namespace();                        // "/unitree2" or "/"
    ns             = (ns == "/") ? "" : ns.substr(1) + "/";  // "unitree2/" or ""
    auto qualify   = [&](std::string& f) {
      if (f.empty()) return;
      if (f.front() == '/') f = f.substr(1);
      else if (!ns.empty()) f = ns + f;
    };
    qualify(map_frame_);
    qualify(odom_frame_);
    qualify(base_frame_);
  }

  prior_map_voxel_size_ = declare_parameter<double>("relocalization.prior_map_voxel_size", 0.4);
  scan_voxel_size_      = declare_parameter<double>("relocalization.scan_voxel_size", 0.4);
  max_corresp_dist_init_ =
      declare_parameter<double>("relocalization.gicp.max_correspondence_distance_init", 5.0);
  max_corresp_dist_track_ =
      declare_parameter<double>("relocalization.gicp.max_correspondence_distance_track", 1.0);
  max_iterations_ = declare_parameter<int>("relocalization.gicp.max_iterations", 50);
  transformation_epsilon_ =
      declare_parameter<double>("relocalization.gicp.transformation_epsilon", 1e-4);
  correction_rate_hz_ = declare_parameter<double>("relocalization.correction_rate_hz", 1.0);
  publish_localized_odom_ =
      declare_parameter<bool>("relocalization.publish_localized_odom", true);

  // Declared for backward compatibility with existing YAML but not used: TF is
  // broadcast on every odometry message so that map->odom carries the same
  // stamp as the pose it corrects, which is strictly better than a free-running
  // timer. Kept declared so old config files do not need editing.
  (void)declare_parameter<double>("relocalization.tf_publish_rate_hz", 50.0);

  // Submap cropping (fix for the whole-map GICP target).
  submap_radius_ = declare_parameter<double>("relocalization.submap.radius", 60.0);
  submap_refresh_move_ =
      declare_parameter<double>("relocalization.submap.refresh_move", 10.0);

  // Sliding-window accumulation.
  accum_cfg_.max_scans = static_cast<std::size_t>(
      std::max<int64_t>(1, declare_parameter<int>("relocalization.accumulator.max_scans", 10)));
  accum_cfg_.max_span_m =
      declare_parameter<double>("relocalization.accumulator.max_span", 20.0);
  accum_cfg_.min_keyframe_move_m =
      declare_parameter<double>("relocalization.accumulator.min_keyframe_move", 0.5);
  accum_cfg_.voxel_size = scan_voxel_size_;

  // Correction gating.
  correction_policy_.alpha = declare_parameter<double>("relocalization.correction.alpha", 0.3);
  correction_policy_.max_translation_step_m =
      declare_parameter<double>("relocalization.correction.max_translation_step", 0.5);
  correction_policy_.max_rotation_step_rad =
      declare_parameter<double>("relocalization.correction.max_rotation_step_deg", 5.0) * kDeg;
  correction_policy_.min_inlier_ratio =
      declare_parameter<double>("relocalization.correction.min_inlier_ratio", 0.4);
  verify_inlier_dist_ =
      declare_parameter<double>("relocalization.correction.inlier_dist", 0.5);
  max_eval_samples_ = static_cast<std::size_t>(
      std::max<int64_t>(0, declare_parameter<int>("relocalization.correction.max_eval_samples", 5000)));

  // Global relocalization.
  global_enabled_ = declare_parameter<bool>("relocalization.global.enabled", true);
  global_attempt_period_s_ =
      declare_parameter<double>("relocalization.global.attempt_period", 3.0);
  global_min_span_m_ = declare_parameter<double>("relocalization.global.min_span", 0.0);
  global_map_voxel_size_ =
      declare_parameter<double>("relocalization.global.map_voxel_size", 0.5);
  global_cfg_.voxel_size = static_cast<float>(
      declare_parameter<double>("relocalization.global.voxel_size", 0.5));
  global_cfg_.use_quatro = declare_parameter<bool>("relocalization.global.use_quatro", true);
  global_cfg_.use_ratio_test =
      declare_parameter<bool>("relocalization.global.use_ratio_test", true);
  global_cfg_.min_final_inliers = static_cast<std::size_t>(
      std::max<int64_t>(1, declare_parameter<int>("relocalization.global.min_final_inliers", 20)));
  global_cfg_.min_source_points = static_cast<std::size_t>(
      std::max<int64_t>(1, declare_parameter<int>("relocalization.global.min_source_points", 2000)));
  global_cfg_.min_verify_inlier_ratio =
      declare_parameter<double>("relocalization.global.min_verify_inlier_ratio", 0.5);
  global_cfg_.required_confirmations =
      declare_parameter<int>("relocalization.global.required_confirmations", 2);
  global_cfg_.max_confirm_disagreement_m =
      declare_parameter<double>("relocalization.global.max_confirm_disagreement", 1.0);
  global_cfg_.max_confirm_disagreement_rad =
      declare_parameter<double>("relocalization.global.max_confirm_disagreement_deg", 6.0) * kDeg;

  lost_recovery_enabled_ =
      declare_parameter<bool>("relocalization.lost_recovery.enabled", true);
  max_consecutive_failures_ =
      declare_parameter<int>("relocalization.lost_recovery.max_consecutive_failures", 15);

  coverage_gate_enabled_ =
      declare_parameter<bool>("relocalization.coverage_gate_enabled", false);
  coverage_radius_   = declare_parameter<double>("relocalization.coverage_radius", 15.0);
  trajectory_file_   = declare_parameter<std::string>("relocalization.trajectory_file", "");
  trajectory_format_ = declare_parameter<std::string>("relocalization.trajectory_format", "auto");

  const auto initialpose_topic =
      declare_parameter<std::string>("relocalization.initialpose_topic", "/initialpose");
  const auto cloud_topic =
      declare_parameter<std::string>("topics.cloud_registered", "fast_lio/cloud_registered");
  const auto odom_topic = declare_parameter<std::string>("topics.odometry", "fast_lio/odometry");
  const auto prior_map_topic =
      declare_parameter<std::string>("relocalization.prior_map_topic", "/prior_map");
  const auto prior_path_topic =
      declare_parameter<std::string>("relocalization.prior_path_topic", "fast_lio/prior_path");
  const auto localized_odom_topic = declare_parameter<std::string>(
      "relocalization.localized_odom_topic", "/localized_odometry");
  const auto status_topic = declare_parameter<std::string>("relocalization.status_topic",
                                                           "fast_lio/relocalization_status");
  const auto diagnostic_topic = declare_parameter<std::string>(
      "relocalization.diagnostic_topic", "fast_lio/relocalization_diagnostic");

  // --- Prior map ----------------------------------------------------------
  if (map_file_.empty()) {
    RCLCPP_FATAL(get_logger(), "relocalization.map_file is empty");
    throw std::runtime_error("relocalization.map_file is empty");
  }
  PointCloudT::Ptr raw(new PointCloudT);
  if (pcl::io::loadPCDFile<PointT>(map_file_, *raw) < 0) {
    RCLCPP_FATAL(get_logger(), "Failed to load prior map: %s", map_file_.c_str());
    throw std::runtime_error("Failed to load prior map");
  }
  RCLCPP_INFO(get_logger(), "Loaded prior map %s with %zu points", map_file_.c_str(), raw->size());

  prior_map_ds_ = reloc::voxelDownsample(PointCloudT::ConstPtr(raw), prior_map_voxel_size_);
  RCLCPP_INFO(get_logger(), "Tracking target: %zu points (leaf=%.3f)", prior_map_ds_->size(),
              prior_map_voxel_size_);
  if (prior_map_ds_->empty()) {
    RCLCPP_FATAL(get_logger(), "Prior map is empty after downsampling");
    throw std::runtime_error("Prior map is empty after downsampling");
  }

  // KISS-Matcher voxelizes internally, but pre-thinning the map keeps the FPFH
  // stage from re-processing millions of points on every global attempt.
  if (global_enabled_) {
    PointCloudT::Ptr global_map =
        reloc::voxelDownsample(PointCloudT::ConstPtr(raw), global_map_voxel_size_);
    prior_map_global_vec_ = reloc::toVector(*global_map);
    RCLCPP_INFO(get_logger(), "Global-registration target: %zu points (leaf=%.3f)",
                prior_map_global_vec_.size(), global_map_voxel_size_);
  }
  raw.reset();

  accumulator_    = std::make_unique<reloc::ScanAccumulator>(accum_cfg_);
  track_cropper_  = std::make_unique<reloc::SubmapCropper>(prior_map_ds_, submap_radius_,
                                                           submap_refresh_move_);
  global_cropper_ = std::make_unique<reloc::SubmapCropper>(prior_map_ds_, submap_radius_,
                                                           /*refresh_move_m=*/0.0);

  if (global_enabled_) {
#ifdef SPARK_FAST_LIO_HAS_KISS_MATCHER
    global_localizer_ = std::make_unique<reloc::GlobalLocalizer>(
        global_cfg_, std::make_shared<reloc::KissMatcherBackend>(global_cfg_));
    RCLCPP_INFO(get_logger(),
                "Global relocalization ENABLED (kiss_matcher, voxel=%.2f, quatro=%s). "
                "No operator initial pose required.",
                global_cfg_.voxel_size, global_cfg_.use_quatro ? "true" : "false");
#else
    global_enabled_ = false;
    RCLCPP_WARN(get_logger(),
                "relocalization.global.enabled is true but this package was built without "
                "KISS-Matcher. Falling back to operator-supplied /initialpose. Build with "
                "kiss_matcher available to enable automatic relocalization.");
#endif
  }

  // --- Prior trajectory ---------------------------------------------------
  {
    const std::string traj =
        trajectory_file_.empty() ? autoDetectTrajectory(map_file_) : trajectory_file_;
    if (!traj.empty()) parseTrajectory(traj, trajectory_format_, traj_positions_);
    if (coverage_gate_enabled_ && traj_positions_.empty()) {
      RCLCPP_WARN(get_logger(),
                  "coverage_gate_enabled but no trajectory found next to the map; gate OFF.");
      coverage_gate_enabled_ = false;
    } else if (!traj_positions_.empty()) {
      RCLCPP_INFO(get_logger(), "Loaded prior trajectory: %zu pts from %s (gate %s).",
                  traj_positions_.size(), traj.c_str(), coverage_gate_enabled_ ? "ON" : "OFF");
    }
  }

  // --- Publishers ---------------------------------------------------------
  rclcpp::QoS latched_qos(1);
  latched_qos.transient_local().reliable();
  pub_prior_map_ = create_publisher<sensor_msgs::msg::PointCloud2>(prior_map_topic, latched_qos);
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*prior_map_ds_, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp    = now();
    pub_prior_map_->publish(msg);
  }
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
  }
  if (publish_localized_odom_) {
    pub_localized_odom_ =
        create_publisher<nav_msgs::msg::Odometry>(localized_odom_topic, rclcpp::QoS(20));
  }
  pub_status_     = create_publisher<std_msgs::msg::Bool>(status_topic, latched_qos);
  pub_diagnostic_ = create_publisher<std_msgs::msg::String>(diagnostic_topic, rclcpp::QoS(10));
  publishStatus(false);

  // --- Subscriptions ------------------------------------------------------
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

  configureGicp(track_gicp_, max_corresp_dist_track_);
  configureGicp(global_gicp_, max_corresp_dist_init_);

  correction_thread_ = std::thread(&Relocalization::correctionLoop, this);
  if (global_enabled_) global_thread_ = std::thread(&Relocalization::globalLoop, this);

  RCLCPP_INFO(get_logger(), "Relocalization node ready in frame '%s' (%s).", map_frame_.c_str(),
              global_enabled_ ? "searching for global pose" : "awaiting /initialpose");
}

Relocalization::~Relocalization() {
  stop_thread_.store(true);
  if (correction_thread_.joinable()) correction_thread_.join();
  if (global_thread_.joinable()) global_thread_.join();
}

void Relocalization::configureGicp(pcl::GeneralizedIterativeClosestPoint<PointT, PointT>& gicp,
                                   double max_corresp_dist) const {
  gicp.setMaxCorrespondenceDistance(max_corresp_dist);
  gicp.setMaximumIterations(max_iterations_);
  gicp.setTransformationEpsilon(transformation_epsilon_);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void Relocalization::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  PointCloudT cloud;
  pcl::fromROSMsg(*msg, cloud);

  std::lock_guard<std::mutex> lk(mutex_);
  if (!has_odom_) return;  // no pose to key the scan to yet
  accumulator_->add(cloud, latest_odom_T_body_.translation());
  has_scan_ = !accumulator_->empty();
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
  RCLCPP_INFO(get_logger(), "Received operator initial pose in frame '%s'",
              msg->header.frame_id.c_str());

  const Snapshot snap = snapshot();
  if (!snap.valid) {
    RCLCPP_WARN(get_logger(), "No scan/odom yet. Ignoring initial pose.");
    return;
  }

  const Eigen::Isometry3d map_T_body_guess = poseMsgToIso(msg->pose.pose);
  const Eigen::Isometry3d map_T_odom_guess = map_T_body_guess * snap.odom_T_body.inverse();

  RefineResult r;
  {
    std::lock_guard<std::mutex> reg_lk(global_reg_mutex_);
    bool target_changed = false;
    r = refine(snap.cloud, map_T_odom_guess, snap.odom_T_body, max_corresp_dist_init_,
               global_gicp_, *global_cropper_, target_changed);
  }

  if (!r.converged || r.quality.inlier_ratio < global_cfg_.min_verify_inlier_ratio) {
    RCLCPP_WARN(get_logger(),
                "Operator initial pose did not verify (converged=%d, inlier_ratio=%.3f < %.3f). "
                "Keeping previous state.",
                static_cast<int>(r.converged), r.quality.inlier_ratio,
                global_cfg_.min_verify_inlier_ratio);
    return;
  }

  {
    std::lock_guard<std::mutex> lk(mutex_);
    map_T_odom_ = r.map_T_odom;
  }
  // An operator pose is an explicit human assertion, so it bypasses the
  // multi-proposal confirmation the automatic search has to satisfy.
  if (global_localizer_) global_localizer_->reset();
  consecutive_failures_.store(0);
  relocalized_.store(true);
  publishStatus(true);
  RCLCPP_INFO(get_logger(), "Relocalized from operator pose. inlier_ratio=%.3f err=%.3f m",
              r.quality.inlier_ratio, r.quality.mean_inlier_error);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

Relocalization::Snapshot Relocalization::snapshot() const {
  Snapshot s;
  std::lock_guard<std::mutex> lk(mutex_);
  if (!has_scan_ || !has_odom_ || accumulator_->empty()) return s;
  s.cloud       = accumulator_->merged();
  s.odom_T_body = latest_odom_T_body_;
  s.map_T_odom  = map_T_odom_;
  s.span_m      = accumulator_->spanMetres();
  s.valid       = s.cloud && !s.cloud->empty();
  return s;
}

Relocalization::RefineResult Relocalization::refine(
    const PointCloudT::ConstPtr& scan_in_odom,
    const Eigen::Isometry3d& map_T_odom_guess,
    const Eigen::Isometry3d& odom_T_body,
    double max_corresp_dist,
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT>& gicp,
    reloc::SubmapCropper& cropper,
    bool& target_changed) {
  RefineResult out;
  out.map_T_odom = map_T_odom_guess;
  target_changed = false;

  if (!scan_in_odom || scan_in_odom->empty()) return out;
  if (!reloc::isFinite(map_T_odom_guess)) return out;

  // Centre the crop on where the guess puts the robot, not on the map origin.
  const Eigen::Vector3d center = (map_T_odom_guess * odom_T_body).translation();
  target_changed               = cropper.update(center);
  if (!cropper.valid()) {
    // Off the edge of the prior map: there is no target to align against.
    return out;
  }

  if (target_changed) {
    gicp.setInputTarget(cropper.submap());
  }
  gicp.setMaxCorrespondenceDistance(max_corresp_dist);
  gicp.setInputSource(scan_in_odom);

  PointCloudT aligned;
  gicp.align(aligned, map_T_odom_guess.matrix().cast<float>());
  if (!gicp.hasConverged()) return out;

  Eigen::Isometry3d T;
  T.matrix() = gicp.getFinalTransformation().cast<double>();
  if (!reloc::isFinite(T)) return out;

  out.map_T_odom = reloc::orthonormalized(T);
  out.quality    = reloc::evaluate(aligned, cropper.tree(), verify_inlier_dist_,
                                   max_eval_samples_);
  out.converged  = true;
  return out;
}

// ---------------------------------------------------------------------------
// Global relocalization loop
// ---------------------------------------------------------------------------

void Relocalization::globalLoop() {
  const auto period = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::duration<double>(std::max(0.1, global_attempt_period_s_)));

  while (!stop_thread_.load() && rclcpp::ok()) {
    std::this_thread::sleep_for(period);
    if (stop_thread_.load()) break;
    if (relocalized_.load() || !global_localizer_) continue;

    const Snapshot snap = snapshot();
    if (!snap.valid) continue;
    if (global_min_span_m_ > 0.0 && snap.span_m < global_min_span_m_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Waiting for motion before global search (%.1f / %.1f m).",
                           snap.span_m, global_min_span_m_);
      continue;
    }

    const auto proposal = global_localizer_->propose(*snap.cloud, prior_map_global_vec_);
    if (!proposal.ok()) {
      RCLCPP_INFO(get_logger(), "Global attempt %zu: %s (inliers=%zu, %.2f s)",
                  global_localizer_->attemptCount(), reloc::toString(proposal.status),
                  proposal.registration.num_final_inliers, proposal.registration.seconds);
      publishDiagnostic(std::string("global:") + reloc::toString(proposal.status));
      continue;
    }

    RefineResult r;
    {
      std::lock_guard<std::mutex> reg_lk(global_reg_mutex_);
      bool target_changed = false;
      r = refine(snap.cloud, proposal.map_T_odom, snap.odom_T_body, max_corresp_dist_init_,
                 global_gicp_, *global_cropper_, target_changed);
    }
    if (!r.converged) {
      RCLCPP_INFO(get_logger(), "Global attempt %zu: proposal did not refine.",
                  global_localizer_->attemptCount());
      publishDiagnostic("global:refine-failed");
      continue;
    }

    const auto v = global_localizer_->submitVerification(r.map_T_odom, r.quality);
    RCLCPP_INFO(get_logger(),
                "Global attempt %zu: %s (kiss_inliers=%zu, %.2f s, verify_ratio=%.3f, "
                "err=%.3f m, confirmations=%d/%d)",
                global_localizer_->attemptCount(), reloc::toString(v.status),
                proposal.registration.num_final_inliers, proposal.registration.seconds,
                r.quality.inlier_ratio, r.quality.mean_inlier_error, v.confirmations,
                global_cfg_.required_confirmations);
    publishDiagnostic(std::string("global:") + reloc::toString(v.status));

    if (!v.confirmed()) continue;

    {
      std::lock_guard<std::mutex> lk(mutex_);
      map_T_odom_ = v.map_T_odom;
    }
    consecutive_failures_.store(0);
    relocalized_.store(true);
    publishStatus(true);

    const Eigen::Vector3d p = (v.map_T_odom * snap.odom_T_body).translation();
    RCLCPP_INFO(get_logger(),
                "RELOCALIZED without operator input at map position (%.2f, %.2f, %.2f).",
                p.x(), p.y(), p.z());
  }
}

// ---------------------------------------------------------------------------
// Tracking loop
// ---------------------------------------------------------------------------

void Relocalization::correctionLoop() {
  const auto period = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::duration<double>(1.0 / std::max(1e-3, correction_rate_hz_)));

  while (!stop_thread_.load() && rclcpp::ok()) {
    std::this_thread::sleep_for(period);
    if (stop_thread_.load()) break;
    if (!relocalized_.load()) continue;

    const Snapshot snap = snapshot();
    if (!snap.valid) continue;

    // Coverage gate: outside the mapped area, scan-to-prior-map GICP has no
    // valid target and would corrupt the correction. Freeze map<-odom (odom
    // dead-reckons the robot forward) and resume once back inside.
    if (coverage_gate_enabled_) {
      const Eigen::Vector3d p_map = (snap.map_T_odom * snap.odom_T_body).translation();
      const bool inside           = insideCoverage(p_map);
      if (inside != inside_coverage_.exchange(inside)) {
        RCLCPP_INFO(get_logger(),
                    inside ? "Back inside prior-map coverage; resuming correction."
                           : "Left prior-map coverage; pausing correction (odom dead-reckons).");
      }
      if (!inside) continue;
    }

    bool target_changed = false;
    const RefineResult r = refine(snap.cloud, snap.map_T_odom, snap.odom_T_body,
                                  max_corresp_dist_track_, track_gicp_, *track_cropper_,
                                  target_changed);

    reloc::CorrectionResult c;
    if (!r.converged) {
      c.verdict = CorrectionVerdict::kRejectedQuality;
      c.pose    = snap.map_T_odom;
    } else {
      c = reloc::applyCorrection(snap.map_T_odom, r.map_T_odom, r.quality, correction_policy_);
    }

    if (c.accepted()) {
      consecutive_failures_.store(0);
      std::lock_guard<std::mutex> lk(mutex_);
      // Re-read under the lock: the global thread or an operator pose may have
      // replaced map_T_odom_ while this cycle was registering. Applying a
      // correction computed against a superseded pose would undo that.
      const auto fresh = reloc::applyCorrection(map_T_odom_, r.map_T_odom, r.quality,
                                                correction_policy_);
      if (fresh.accepted()) map_T_odom_ = fresh.pose;
      RCLCPP_DEBUG(get_logger(), "Tracking OK. ratio=%.3f err=%.3f m d=%.3f m/%.2f deg",
                   r.quality.inlier_ratio, r.quality.mean_inlier_error, c.delta.translation_m,
                   c.delta.rotation_rad / kDeg);
    } else {
      const int fails = consecutive_failures_.fetch_add(1) + 1;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Tracking %s (ratio=%.3f, d=%.3f m/%.2f deg). %d consecutive.",
                           reloc::toString(c.verdict), r.quality.inlier_ratio,
                           c.delta.translation_m, c.delta.rotation_rad / kDeg, fails);
      publishDiagnostic(std::string("track:") + reloc::toString(c.verdict));

      if (lost_recovery_enabled_ && max_consecutive_failures_ > 0 &&
          fails >= max_consecutive_failures_) {
        RCLCPP_ERROR(get_logger(),
                     "Lost relocalization after %d consecutive failed corrections. "
                     "Dropping back to %s.",
                     fails, global_enabled_ ? "global search" : "awaiting /initialpose");
        relocalized_.store(false);
        consecutive_failures_.store(0);
        if (global_localizer_) global_localizer_->reset();
        publishStatus(false);
        publishDiagnostic("track:lost");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

bool Relocalization::insideCoverage(const Eigen::Vector3d& p_map) const {
  if (traj_positions_.empty()) return true;
  double best_sq = std::numeric_limits<double>::max();
  for (const auto& t : traj_positions_) best_sq = std::min(best_sq, (t - p_map).squaredNorm());
  return best_sq <= coverage_radius_ * coverage_radius_;
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

void Relocalization::publishStatus(bool relocalized) {
  if (!pub_status_) return;
  std_msgs::msg::Bool b;
  b.data = relocalized;
  pub_status_->publish(b);
}

void Relocalization::publishDiagnostic(const std::string& text) {
  if (!pub_diagnostic_) return;
  std_msgs::msg::String s;
  s.data = text;
  pub_diagnostic_->publish(s);
}

}  // namespace spark_fast_lio

RCLCPP_COMPONENTS_REGISTER_NODE(spark_fast_lio::Relocalization)
