#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "global_localizer.h"
#include "reloc_core.h"

namespace spark_fast_lio {

// Localizes FAST-LIO odometry against a prior .pcd map.
//
// Two loops run concurrently:
//
//   * The global loop owns startup. It accumulates a short window of scans and
//     hands them to KISS-Matcher for correspondence-based registration against
//     the prior map, which needs no initial guess. A proposal must survive a
//     GICP refinement and then agree with a second, independent proposal before
//     it latches. An operator pose on /initialpose still works and simply
//     bypasses the search.
//
//   * The tracking loop owns steady state. It aligns the accumulated window
//     against a cropped submap of the prior map and folds the result into
//     map<-odom through a quality- and step-gated interpolation.
//
// Only map<-odom is ever estimated, never map<-base_link: the source cloud is
// already in the odom frame, so the estimate stays valid even when registration
// takes longer than one cycle and the robot has moved on.
class Relocalization : public rclcpp::Node {
 public:
  using PointT      = reloc::PointT;
  using PointCloudT = reloc::PointCloudT;

  explicit Relocalization(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~Relocalization() override;

 private:
  // --- Callbacks -----------------------------------------------------------
  void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);

  // --- Threads -------------------------------------------------------------
  void correctionLoop();
  void globalLoop();

  // --- Registration --------------------------------------------------------
  struct RefineResult {
    bool converged{false};
    Eigen::Isometry3d map_T_odom{Eigen::Isometry3d::Identity()};
    reloc::RegistrationQuality quality;
  };

  // Aligns `scan_in_odom` against a crop of the prior map centred on the pose
  // implied by `map_T_odom_guess`, and scores the result. `gicp` and `cropper`
  // are passed in so the tracking and global threads keep separate, non-shared
  // registration state (PCL's GICP is not reentrant).
  RefineResult refine(const PointCloudT::ConstPtr& scan_in_odom,
                      const Eigen::Isometry3d& map_T_odom_guess,
                      const Eigen::Isometry3d& odom_T_body,
                      double max_corresp_dist,
                      pcl::GeneralizedIterativeClosestPoint<PointT, PointT>& gicp,
                      reloc::SubmapCropper& cropper,
                      bool& target_changed);

  // --- Output --------------------------------------------------------------
  void broadcastMapToOdomTf(const rclcpp::Time& stamp);
  void publishLocalizedOdom(const rclcpp::Time& stamp);
  void publishStatus(bool relocalized);
  void publishDiagnostic(const std::string& text);

  bool insideCoverage(const Eigen::Vector3d& p_map) const;

  // Snapshot of the accumulated window plus the odom pose that goes with it.
  struct Snapshot {
    PointCloudT::Ptr cloud;
    Eigen::Isometry3d odom_T_body{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d map_T_odom{Eigen::Isometry3d::Identity()};
    double span_m{0.0};
    bool valid{false};
  };
  Snapshot snapshot() const;

  void configureGicp(pcl::GeneralizedIterativeClosestPoint<PointT, PointT>& gicp,
                     double max_corresp_dist) const;

  // --- Parameters ----------------------------------------------------------
  std::string map_file_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string base_frame_;

  double prior_map_voxel_size_{0.4};
  double scan_voxel_size_{0.4};
  double max_corresp_dist_init_{5.0};
  double max_corresp_dist_track_{1.0};
  int max_iterations_{50};
  double transformation_epsilon_{1e-4};
  double correction_rate_hz_{1.0};
  bool publish_localized_odom_{true};

  double submap_radius_{60.0};
  double submap_refresh_move_{10.0};

  reloc::AccumulatorConfig accum_cfg_;
  reloc::CorrectionPolicy correction_policy_;
  double verify_inlier_dist_{0.5};
  std::size_t max_eval_samples_{5000};

  bool global_enabled_{true};
  double global_attempt_period_s_{3.0};
  double global_min_span_m_{0.0};
  double global_map_voxel_size_{0.5};
  reloc::GlobalLocalizerConfig global_cfg_;

  bool lost_recovery_enabled_{true};
  int max_consecutive_failures_{15};

  bool coverage_gate_enabled_{false};
  double coverage_radius_{15.0};
  std::string trajectory_file_;
  std::string trajectory_format_{"auto"};
  std::vector<Eigen::Vector3d> traj_positions_;
  std::atomic<bool> inside_coverage_{true};

  // --- State ---------------------------------------------------------------
  mutable std::mutex mutex_;
  Eigen::Isometry3d map_T_odom_{Eigen::Isometry3d::Identity()};
  std::atomic<bool> relocalized_{false};
  std::atomic<int> consecutive_failures_{0};

  PointCloudT::Ptr prior_map_ds_;                      // tracking target
  std::vector<Eigen::Vector3f> prior_map_global_vec_;  // KISS-Matcher target

  std::unique_ptr<reloc::ScanAccumulator> accumulator_;
  std::unique_ptr<reloc::SubmapCropper> track_cropper_;
  std::unique_ptr<reloc::SubmapCropper> global_cropper_;
  std::unique_ptr<reloc::GlobalLocalizer> global_localizer_;

  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> track_gicp_;
  // Guards global_gicp_/global_cropper_, which are reachable both from the
  // global thread and from the /initialpose subscriber callback. PCL's GICP
  // holds mutable per-alignment state and is not reentrant.
  std::mutex global_reg_mutex_;
  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> global_gicp_;

  Eigen::Isometry3d latest_odom_T_body_{Eigen::Isometry3d::Identity()};
  rclcpp::Time latest_odom_stamp_;
  bool has_scan_{false};
  bool has_odom_{false};

  // --- ROS interfaces ------------------------------------------------------
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_initialpose_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_prior_map_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_prior_path_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localized_odom_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_status_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_diagnostic_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::thread correction_thread_;
  std::thread global_thread_;
  std::atomic<bool> stop_thread_{false};
};

}  // namespace spark_fast_lio
