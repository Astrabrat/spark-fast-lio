// ROS-free geometry and policy helpers for prior-map relocalization.
//
// Everything here is deliberately free of rclcpp so it can be unit tested
// without a ROS graph. relocalization.cpp is the only ROS-facing consumer.
#pragma once

#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace spark_fast_lio {
namespace reloc {

using PointT      = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

// ---------------------------------------------------------------------------
// SE(3) utilities
// ---------------------------------------------------------------------------

struct PoseDelta {
  double translation_m{0.0};
  double rotation_rad{0.0};
};

// Magnitude of the relative transform b^-1 * a. Rotation is the geodesic angle
// on SO(3), in [0, pi].
PoseDelta poseDelta(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b);

// SLERP on rotation, LERP on translation. `alpha` is clamped to [0, 1];
// alpha == 0 returns `from`, alpha == 1 returns `to`.
Eigen::Isometry3d interpolate(const Eigen::Isometry3d& from,
                              const Eigen::Isometry3d& to,
                              double alpha);

// Re-projects the linear block onto SO(3). GICP and GNC solutions accumulate
// enough numerical drift over long runs that repeated composition visibly
// scales the transform if this is skipped.
Eigen::Isometry3d orthonormalized(const Eigen::Isometry3d& T);

// True iff every coefficient of the 4x4 matrix is finite.
bool isFinite(const Eigen::Isometry3d& T);

// ---------------------------------------------------------------------------
// Registration quality
// ---------------------------------------------------------------------------

struct RegistrationQuality {
  // Fraction of evaluated source points with a target neighbour within
  // `inlier_dist`. Scale free, so it is safe to threshold directly.
  double inlier_ratio{0.0};
  // Mean Euclidean error over inliers only, in metres.
  double mean_inlier_error{std::numeric_limits<double>::infinity()};
  std::size_t num_inliers{0};
  std::size_t num_evaluated{0};
};

// Bounded alternative to pcl::Registration::getFitnessScore().
//
// getFitnessScore() averages *squared* nearest-neighbour distance over every
// source point with an unbounded search range, so its value moves with how much
// of the scan happens to fall outside the map — the same registration scores
// differently at the map edge than at its centre, which makes an absolute
// threshold on it meaningless. Reporting an inlier ratio plus a mean error over
// inliers separates "how much of the scan matched" from "how well it matched".
//
// `max_samples` > 0 evaluates a uniform stride over the cloud instead of every
// point, which bounds the cost for dense scans.
RegistrationQuality evaluate(const PointCloudT& source_in_map,
                             const pcl::KdTreeFLANN<PointT>& target_tree,
                             double inlier_dist,
                             std::size_t max_samples = 0);

// ---------------------------------------------------------------------------
// Prior-map submap cropping
// ---------------------------------------------------------------------------

// Maintains a radius crop of the prior map around the current pose estimate,
// plus a KD-tree over that crop.
//
// The point is cost: PCL's GICP recomputes target covariances (a 20-NN PCA per
// point) whenever the target changes, so aligning against the whole prior map
// every cycle forces a coarse voxel size to stay real time. Cropping bounds the
// target, and re-cropping only after the centre has moved `refresh_move_m`
// means the steady-state per-cycle cost is one distance comparison.
class SubmapCropper {
 public:
  // `radius_m` <= 0 disables cropping: the whole map is used and update() only
  // ever builds once.
  SubmapCropper(PointCloudT::ConstPtr map, double radius_m, double refresh_move_m);

  // Returns true if this call rebuilt the submap (and therefore the KD-tree),
  // which is the signal the caller needs to re-point GICP at a new target.
  bool update(const Eigen::Vector3d& center);

  PointCloudT::ConstPtr submap() const { return submap_; }
  // Only valid once update() has produced a non-empty submap.
  const pcl::KdTreeFLANN<PointT>& tree() const { return tree_; }

  bool valid() const { return submap_ && !submap_->empty(); }
  const Eigen::Vector3d& lastCenter() const { return last_center_; }
  std::size_t rebuildCount() const { return rebuild_count_; }
  std::size_t mapSize() const { return map_ ? map_->size() : 0; }

 private:
  PointCloudT::ConstPtr map_;
  double radius_m_{0.0};
  double refresh_move_m_{0.0};

  pcl::KdTreeFLANN<PointT> map_tree_;
  bool map_tree_ready_{false};

  PointCloudT::Ptr submap_;
  pcl::KdTreeFLANN<PointT> tree_;
  Eigen::Vector3d last_center_{Eigen::Vector3d::Zero()};
  bool has_center_{false};
  std::size_t rebuild_count_{0};
};

// ---------------------------------------------------------------------------
// Sliding-window scan accumulator
// ---------------------------------------------------------------------------

struct AccumulatorConfig {
  // Hard cap on retained keyframes.
  std::size_t max_scans{10};
  // Drop the oldest keyframes once the retained path exceeds this length.
  // <= 0 disables the span limit.
  double max_span_m{20.0};
  // A scan is only retained if the sensor has moved this far since the last
  // retained one. Without it a stationary robot fills the window with copies of
  // a single viewpoint, which adds cost and no geometric information.
  double min_keyframe_move_m{0.5};
  // Voxel leaf applied to the merged output. <= 0 disables.
  double voxel_size{0.0};
};

// Collects FAST-LIO `cloud_registered` frames (already in the odom frame) into
// a rolling local submap.
//
// A single LiDAR sweep is a thin constraint: it under-determines z, roll and
// pitch in open or self-similar geometry, and gives global registration too
// little structure to key on. Odometry is locally accurate over a few metres
// even while drifting globally, so stacking a short window of scans in the odom
// frame is a nearly free way to get a much better conditioned source cloud.
class ScanAccumulator {
 public:
  explicit ScanAccumulator(AccumulatorConfig cfg = {});

  // Returns true if the scan was retained as a keyframe. Empty scans and
  // non-finite positions are always rejected.
  bool add(const PointCloudT& scan_in_odom, const Eigen::Vector3d& sensor_pos_in_odom);

  // Never null; empty when nothing has been retained.
  PointCloudT::Ptr merged() const;

  std::size_t size() const { return keyframes_.size(); }
  std::size_t pointCount() const;
  // Path length across the retained window, in metres.
  double spanMetres() const;
  bool empty() const { return keyframes_.empty(); }
  void clear();

  const AccumulatorConfig& config() const { return cfg_; }

 private:
  struct Keyframe {
    PointCloudT::Ptr cloud;
    Eigen::Vector3d position;
  };

  void evict();

  AccumulatorConfig cfg_;
  std::deque<Keyframe> keyframes_;
  Eigen::Vector3d last_accepted_pos_{Eigen::Vector3d::Zero()};
  bool has_last_accepted_{false};
};

// ---------------------------------------------------------------------------
// Correction gating
// ---------------------------------------------------------------------------

struct CorrectionPolicy {
  // Fraction of the way to move toward the candidate. 1.0 reproduces the old
  // hard-overwrite behaviour.
  double alpha{0.3};
  // Corrections larger than these are treated as mis-registrations, not as
  // drift that accumulated in one cycle.
  double max_translation_step_m{0.5};
  double max_rotation_step_rad{0.0873};  // 5 degrees
  double min_inlier_ratio{0.4};
};

enum class CorrectionVerdict {
  kAccepted,
  kRejectedQuality,   // too little of the scan matched the map
  kRejectedJump,      // implausibly large single-cycle correction
  kRejectedInvalid,   // non-finite candidate
};

struct CorrectionResult {
  CorrectionVerdict verdict{CorrectionVerdict::kRejectedInvalid};
  // The pose to adopt. On any rejection this is `current`, unchanged.
  Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
  PoseDelta delta;  // magnitude of candidate relative to current
  bool accepted() const { return verdict == CorrectionVerdict::kAccepted; }
};

// Decides whether to fold `candidate` into `current`, and by how much.
//
// Three things are conflated in the naive `current = candidate` update: a good
// correction, a mis-registration, and a discontinuity in the published TF. This
// separates them — quality gates out bad registrations, the step limit gates
// out mis-registrations that still scored well (the aliasing case), and the
// interpolation keeps map->odom continuous for downstream consumers.
CorrectionResult applyCorrection(const Eigen::Isometry3d& current,
                                 const Eigen::Isometry3d& candidate,
                                 const RegistrationQuality& quality,
                                 const CorrectionPolicy& policy);

const char* toString(CorrectionVerdict v);

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

// Drops non-finite points. KISS-Matcher takes bare float vectors.
std::vector<Eigen::Vector3f> toVector(const PointCloudT& cloud);

PointCloudT::Ptr voxelDownsample(const PointCloudT::ConstPtr& in, double leaf);

// Removes non-finite points without downsampling.
PointCloudT::Ptr removeNonFinite(const PointCloudT::ConstPtr& in);

}  // namespace reloc
}  // namespace spark_fast_lio
