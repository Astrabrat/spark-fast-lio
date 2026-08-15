#include "reloc_core.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <pcl/filters/voxel_grid.h>

namespace spark_fast_lio {
namespace reloc {

// ---------------------------------------------------------------------------
// SE(3) utilities
// ---------------------------------------------------------------------------

PoseDelta poseDelta(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
  PoseDelta d;
  const Eigen::Isometry3d rel = b.inverse() * a;
  d.translation_m             = rel.translation().norm();
  Eigen::Quaterniond q(rel.linear());
  q.normalize();
  // 2 * acos(|w|) is the geodesic angle, folded into [0, pi] so that q and -q
  // (the same rotation) give the same answer.
  const double w = std::min(1.0, std::abs(q.w()));
  d.rotation_rad = 2.0 * std::acos(w);
  return d;
}

Eigen::Isometry3d interpolate(const Eigen::Isometry3d& from,
                              const Eigen::Isometry3d& to,
                              double alpha) {
  alpha = std::clamp(alpha, 0.0, 1.0);
  Eigen::Quaterniond q_from(from.linear());
  Eigen::Quaterniond q_to(to.linear());
  q_from.normalize();
  q_to.normalize();
  // Take the short way round.
  if (q_from.dot(q_to) < 0.0) q_to.coeffs() *= -1.0;

  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  out.linear()          = q_from.slerp(alpha, q_to).normalized().toRotationMatrix();
  out.translation()     = (1.0 - alpha) * from.translation() + alpha * to.translation();
  return out;
}

Eigen::Isometry3d orthonormalized(const Eigen::Isometry3d& T) {
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(T.linear(),
                                        Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0.0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1.0;
    R = svd.matrixU() * V.transpose();
  }
  out.linear()      = R;
  out.translation() = T.translation();
  return out;
}

bool isFinite(const Eigen::Isometry3d& T) { return T.matrix().allFinite(); }

// ---------------------------------------------------------------------------
// Registration quality
// ---------------------------------------------------------------------------

RegistrationQuality evaluate(const PointCloudT& source_in_map,
                             const pcl::KdTreeFLANN<PointT>& target_tree,
                             double inlier_dist,
                             std::size_t max_samples) {
  RegistrationQuality q;
  if (source_in_map.empty() || inlier_dist <= 0.0) return q;
  if (target_tree.getInputCloud() == nullptr || target_tree.getInputCloud()->empty()) return q;

  const std::size_t n = source_in_map.size();
  const std::size_t stride =
      (max_samples > 0 && n > max_samples) ? (n + max_samples - 1) / max_samples : 1;

  const float max_sq = static_cast<float>(inlier_dist * inlier_dist);
  std::vector<int> idx(1);
  std::vector<float> sq_dist(1);
  double error_sum = 0.0;

  for (std::size_t i = 0; i < n; i += stride) {
    const PointT& p = source_in_map[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    ++q.num_evaluated;
    if (target_tree.nearestKSearch(p, 1, idx, sq_dist) < 1) continue;
    if (sq_dist[0] > max_sq) continue;
    ++q.num_inliers;
    error_sum += std::sqrt(static_cast<double>(sq_dist[0]));
  }

  if (q.num_evaluated > 0) {
    q.inlier_ratio = static_cast<double>(q.num_inliers) / static_cast<double>(q.num_evaluated);
  }
  if (q.num_inliers > 0) {
    q.mean_inlier_error = error_sum / static_cast<double>(q.num_inliers);
  }
  return q;
}

// ---------------------------------------------------------------------------
// SubmapCropper
// ---------------------------------------------------------------------------

SubmapCropper::SubmapCropper(PointCloudT::ConstPtr map, double radius_m, double refresh_move_m)
    : map_(std::move(map)),
      radius_m_(radius_m),
      refresh_move_m_(std::max(0.0, refresh_move_m)),
      submap_(new PointCloudT) {
  if (map_ && !map_->empty() && radius_m_ > 0.0) {
    map_tree_.setInputCloud(map_);
    map_tree_ready_ = true;
  }
}

bool SubmapCropper::update(const Eigen::Vector3d& center) {
  if (!map_ || map_->empty()) return false;
  if (!center.allFinite()) return false;

  // Cropping disabled: hand back the whole map, once.
  if (radius_m_ <= 0.0) {
    if (rebuild_count_ > 0) return false;
    submap_.reset(new PointCloudT(*map_));
    tree_.setInputCloud(submap_);
    last_center_ = center;
    has_center_  = true;
    ++rebuild_count_;
    return true;
  }

  if (has_center_ && (center - last_center_).norm() <= refresh_move_m_) return false;

  PointT query;
  query.x = static_cast<float>(center.x());
  query.y = static_cast<float>(center.y());
  query.z = static_cast<float>(center.z());
  query.intensity = 0.0f;

  std::vector<int> indices;
  std::vector<float> sq_dists;
  map_tree_.radiusSearch(query, radius_m_, indices, sq_dists);

  PointCloudT::Ptr cropped(new PointCloudT);
  cropped->reserve(indices.size());
  for (const int i : indices) cropped->points.push_back(map_->points[i]);
  cropped->width    = static_cast<uint32_t>(cropped->points.size());
  cropped->height   = 1;
  cropped->is_dense = true;

  submap_ = cropped;
  // A KD-tree over an empty cloud throws in PCL, so only build when populated.
  // Callers must check valid() before touching tree().
  if (!submap_->empty()) tree_.setInputCloud(submap_);

  last_center_ = center;
  has_center_  = true;
  ++rebuild_count_;
  return true;
}

// ---------------------------------------------------------------------------
// ScanAccumulator
// ---------------------------------------------------------------------------

ScanAccumulator::ScanAccumulator(AccumulatorConfig cfg) : cfg_(cfg) {
  if (cfg_.max_scans == 0) cfg_.max_scans = 1;
}

bool ScanAccumulator::add(const PointCloudT& scan_in_odom,
                          const Eigen::Vector3d& sensor_pos_in_odom) {
  if (scan_in_odom.empty()) return false;
  if (!sensor_pos_in_odom.allFinite()) return false;

  if (has_last_accepted_ && cfg_.min_keyframe_move_m > 0.0) {
    const double moved = (sensor_pos_in_odom - last_accepted_pos_).norm();
    if (moved < cfg_.min_keyframe_move_m) return false;
  }

  PointCloudT::Ptr copy(new PointCloudT);
  copy->reserve(scan_in_odom.size());
  for (const auto& p : scan_in_odom.points) {
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) copy->points.push_back(p);
  }
  if (copy->points.empty()) return false;
  copy->width    = static_cast<uint32_t>(copy->points.size());
  copy->height   = 1;
  copy->is_dense = true;

  keyframes_.push_back(Keyframe{copy, sensor_pos_in_odom});
  last_accepted_pos_ = sensor_pos_in_odom;
  has_last_accepted_ = true;
  evict();
  return true;
}

void ScanAccumulator::evict() {
  while (keyframes_.size() > cfg_.max_scans) keyframes_.pop_front();
  if (cfg_.max_span_m > 0.0) {
    while (keyframes_.size() > 1 && spanMetres() > cfg_.max_span_m) keyframes_.pop_front();
  }
}

double ScanAccumulator::spanMetres() const {
  double span = 0.0;
  for (std::size_t i = 1; i < keyframes_.size(); ++i) {
    span += (keyframes_[i].position - keyframes_[i - 1].position).norm();
  }
  return span;
}

std::size_t ScanAccumulator::pointCount() const {
  std::size_t n = 0;
  for (const auto& kf : keyframes_) n += kf.cloud->size();
  return n;
}

PointCloudT::Ptr ScanAccumulator::merged() const {
  PointCloudT::Ptr out(new PointCloudT);
  out->reserve(pointCount());
  for (const auto& kf : keyframes_) *out += *kf.cloud;
  out->width    = static_cast<uint32_t>(out->points.size());
  out->height   = 1;
  out->is_dense = true;
  if (cfg_.voxel_size > 0.0 && !out->empty()) {
    return voxelDownsample(PointCloudT::ConstPtr(out), cfg_.voxel_size);
  }
  return out;
}

void ScanAccumulator::clear() {
  keyframes_.clear();
  has_last_accepted_ = false;
  last_accepted_pos_.setZero();
}

// ---------------------------------------------------------------------------
// Correction gating
// ---------------------------------------------------------------------------

CorrectionResult applyCorrection(const Eigen::Isometry3d& current,
                                 const Eigen::Isometry3d& candidate,
                                 const RegistrationQuality& quality,
                                 const CorrectionPolicy& policy) {
  CorrectionResult r;
  r.pose = current;

  if (!isFinite(candidate)) {
    r.verdict = CorrectionVerdict::kRejectedInvalid;
    return r;
  }

  r.delta = poseDelta(candidate, current);

  if (quality.inlier_ratio < policy.min_inlier_ratio) {
    r.verdict = CorrectionVerdict::kRejectedQuality;
    return r;
  }

  const bool trans_ok = policy.max_translation_step_m <= 0.0 ||
                        r.delta.translation_m <= policy.max_translation_step_m;
  const bool rot_ok =
      policy.max_rotation_step_rad <= 0.0 || r.delta.rotation_rad <= policy.max_rotation_step_rad;
  if (!trans_ok || !rot_ok) {
    r.verdict = CorrectionVerdict::kRejectedJump;
    return r;
  }

  r.verdict = CorrectionVerdict::kAccepted;
  r.pose    = orthonormalized(interpolate(current, candidate, policy.alpha));
  return r;
}

const char* toString(CorrectionVerdict v) {
  switch (v) {
    case CorrectionVerdict::kAccepted:        return "accepted";
    case CorrectionVerdict::kRejectedQuality: return "rejected:quality";
    case CorrectionVerdict::kRejectedJump:    return "rejected:jump";
    case CorrectionVerdict::kRejectedInvalid: return "rejected:invalid";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

std::vector<Eigen::Vector3f> toVector(const PointCloudT& cloud) {
  std::vector<Eigen::Vector3f> out;
  out.reserve(cloud.size());
  for (const auto& p : cloud.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    out.emplace_back(p.x, p.y, p.z);
  }
  return out;
}

PointCloudT::Ptr removeNonFinite(const PointCloudT::ConstPtr& in) {
  PointCloudT::Ptr out(new PointCloudT);
  if (!in) return out;
  out->reserve(in->size());
  for (const auto& p : in->points) {
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) out->points.push_back(p);
  }
  out->width    = static_cast<uint32_t>(out->points.size());
  out->height   = 1;
  out->is_dense = true;
  return out;
}

PointCloudT::Ptr voxelDownsample(const PointCloudT::ConstPtr& in, double leaf) {
  PointCloudT::Ptr clean = removeNonFinite(in);
  if (leaf <= 0.0 || clean->empty()) return clean;
  PointCloudT::Ptr out(new PointCloudT);
  pcl::VoxelGrid<PointT> vg;
  vg.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
  vg.setInputCloud(clean);
  vg.filter(*out);
  return out;
}

}  // namespace reloc
}  // namespace spark_fast_lio
