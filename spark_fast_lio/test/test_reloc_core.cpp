// Unit tests for the ROS-free relocalization primitives.
//
// These target the assumptions the node relies on, and the edge cases that
// would otherwise only show up on the robot: empty/NaN inputs, degenerate
// configuration, eviction boundaries, and the difference between an inlier
// ratio and PCL's unbounded fitness score.

#include "reloc_core.h"

#include <cmath>

#include <gtest/gtest.h>

using namespace spark_fast_lio::reloc;

namespace {

constexpr double kEps = 1e-9;
constexpr double kDeg = M_PI / 180.0;

Eigen::Isometry3d makePose(double x, double y, double z, double yaw_deg,
                           double pitch_deg = 0.0, double roll_deg = 0.0) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = (Eigen::AngleAxisd(yaw_deg * kDeg, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch_deg * kDeg, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_deg * kDeg, Eigen::Vector3d::UnitX()))
                   .toRotationMatrix();
  T.translation() = Eigen::Vector3d(x, y, z);
  return T;
}

// A dense-ish axis-aligned grid of points, so KD-tree queries have real
// neighbours at known spacing.
PointCloudT::Ptr makeGrid(double x0, double x1, double y0, double y1, double step,
                          double z = 0.0) {
  PointCloudT::Ptr c(new PointCloudT);
  for (double x = x0; x <= x1 + kEps; x += step) {
    for (double y = y0; y <= y1 + kEps; y += step) {
      PointT p;
      p.x = static_cast<float>(x);
      p.y = static_cast<float>(y);
      p.z = static_cast<float>(z);
      p.intensity = 1.0f;
      c->points.push_back(p);
    }
  }
  c->width    = static_cast<uint32_t>(c->points.size());
  c->height   = 1;
  c->is_dense = true;
  return c;
}

PointT pt(float x, float y, float z) {
  PointT p;
  p.x = x;
  p.y = y;
  p.z = z;
  p.intensity = 0.0f;
  return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// poseDelta
// ---------------------------------------------------------------------------

TEST(PoseDelta, IdenticalPosesHaveZeroDelta) {
  const auto T = makePose(3.0, -2.0, 1.0, 47.0);
  const auto d = poseDelta(T, T);
  EXPECT_NEAR(d.translation_m, 0.0, 1e-12);
  EXPECT_NEAR(d.rotation_rad, 0.0, 1e-6);
}

TEST(PoseDelta, PureTranslationIsMeasuredInTheBodyFrame) {
  // b^-1 * a, so a 3-4-0 offset expressed in b's frame still has norm 5
  // regardless of b's orientation.
  const auto b = makePose(0.0, 0.0, 0.0, 90.0);
  const auto a = makePose(3.0, 4.0, 0.0, 90.0);
  const auto d = poseDelta(a, b);
  EXPECT_NEAR(d.translation_m, 5.0, 1e-9);
  EXPECT_NEAR(d.rotation_rad, 0.0, 1e-6);
}

TEST(PoseDelta, RotationIsTheGeodesicAngle) {
  const auto a = makePose(0, 0, 0, 30.0);
  const auto b = makePose(0, 0, 0, -20.0);
  const auto d = poseDelta(a, b);
  EXPECT_NEAR(d.rotation_rad, 50.0 * kDeg, 1e-9);
}

TEST(PoseDelta, RotationFoldsIntoZeroToPi) {
  // 350 degrees apart is 10 degrees the other way; the metric must not report
  // the long way round or the jump gate would fire on tiny corrections near
  // the wrap point.
  const auto a = makePose(0, 0, 0, 355.0);
  const auto b = makePose(0, 0, 0, 5.0);
  const auto d = poseDelta(a, b);
  EXPECT_NEAR(d.rotation_rad, 10.0 * kDeg, 1e-9);
  EXPECT_LE(d.rotation_rad, M_PI + kEps);
}

TEST(PoseDelta, IsSymmetricInMagnitude) {
  const auto a = makePose(1.0, 2.0, 3.0, 25.0, 10.0);
  const auto b = makePose(-1.0, 0.5, 2.0, -40.0, 5.0);
  const auto ab = poseDelta(a, b);
  const auto ba = poseDelta(b, a);
  EXPECT_NEAR(ab.rotation_rad, ba.rotation_rad, 1e-9);
  EXPECT_NEAR(ab.translation_m, ba.translation_m, 1e-9);
}

// ---------------------------------------------------------------------------
// interpolate / orthonormalized
// ---------------------------------------------------------------------------

TEST(Interpolate, EndpointsAreExact) {
  const auto a = makePose(0, 0, 0, 0.0);
  const auto b = makePose(10, 5, 2, 80.0);

  const auto at0 = interpolate(a, b, 0.0);
  EXPECT_NEAR(poseDelta(at0, a).translation_m, 0.0, 1e-9);
  EXPECT_NEAR(poseDelta(at0, a).rotation_rad, 0.0, 1e-7);

  const auto at1 = interpolate(a, b, 1.0);
  EXPECT_NEAR(poseDelta(at1, b).translation_m, 0.0, 1e-9);
  EXPECT_NEAR(poseDelta(at1, b).rotation_rad, 0.0, 1e-7);
}

TEST(Interpolate, HalfwayIsHalfway) {
  const auto a = makePose(0, 0, 0, 0.0);
  const auto b = makePose(10, 0, 0, 90.0);
  const auto m = interpolate(a, b, 0.5);
  EXPECT_NEAR(m.translation().x(), 5.0, 1e-9);
  EXPECT_NEAR(poseDelta(m, a).rotation_rad, 45.0 * kDeg, 1e-7);
}

TEST(Interpolate, AlphaIsClamped) {
  const auto a = makePose(0, 0, 0, 0.0);
  const auto b = makePose(10, 0, 0, 90.0);
  EXPECT_NEAR(interpolate(a, b, 5.0).translation().x(), 10.0, 1e-9);
  EXPECT_NEAR(interpolate(a, b, -3.0).translation().x(), 0.0, 1e-9);
}

TEST(Interpolate, TakesTheShortPathAcrossTheQuaternionSignFlip) {
  // 170 -> -170 degrees is 20 degrees apart. Without the dot-product sign fix
  // SLERP would sweep 340 degrees the other way.
  const auto a = makePose(0, 0, 0, 170.0);
  const auto b = makePose(0, 0, 0, -170.0);
  const auto m = interpolate(a, b, 0.5);
  EXPECT_NEAR(poseDelta(m, a).rotation_rad, 10.0 * kDeg, 1e-7);
  EXPECT_NEAR(poseDelta(m, b).rotation_rad, 10.0 * kDeg, 1e-7);
}

TEST(Interpolate, ResultIsAProperRotation) {
  const auto a = makePose(0, 0, 0, 12.0, 30.0, -20.0);
  const auto b = makePose(1, 2, 3, 200.0, -40.0, 70.0);
  for (double alpha : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const auto m = interpolate(a, b, alpha);
    EXPECT_NEAR(m.linear().determinant(), 1.0, 1e-9) << "alpha=" << alpha;
    EXPECT_TRUE((m.linear() * m.linear().transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-9));
  }
}

TEST(Orthonormalized, RepairsAScaledRotationAndKeepsTranslation) {
  Eigen::Isometry3d T = makePose(4.0, -1.0, 0.5, 33.0);
  T.linear() *= 1.05;  // the drift GICP/GNC output accumulates
  ASSERT_GT(std::abs(T.linear().determinant() - 1.0), 1e-3);

  const auto fixed = orthonormalized(T);
  EXPECT_NEAR(fixed.linear().determinant(), 1.0, 1e-9);
  EXPECT_TRUE((fixed.linear() * fixed.linear().transpose())
                  .isApprox(Eigen::Matrix3d::Identity(), 1e-9));
  EXPECT_TRUE(fixed.translation().isApprox(T.translation()));
}

TEST(Orthonormalized, NeverProducesAReflection) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear()          = -Eigen::Matrix3d::Identity();  // det = -1
  EXPECT_NEAR(orthonormalized(T).linear().determinant(), 1.0, 1e-9);
}

TEST(IsFinite, DetectsNaNAndInf) {
  auto T = makePose(1, 2, 3, 10.0);
  EXPECT_TRUE(isFinite(T));
  T.translation().y() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(isFinite(T));

  auto U = makePose(1, 2, 3, 10.0);
  U.linear()(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(isFinite(U));
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

class EvaluateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    map_ = makeGrid(0.0, 10.0, 0.0, 10.0, 0.5);
    tree_.setInputCloud(map_);
  }
  PointCloudT::Ptr map_;
  pcl::KdTreeFLANN<PointT> tree_;
};

TEST_F(EvaluateTest, PerfectOverlapScoresOne) {
  const auto q = evaluate(*map_, tree_, 0.1);
  EXPECT_EQ(q.num_evaluated, map_->size());
  EXPECT_NEAR(q.inlier_ratio, 1.0, 1e-12);
  EXPECT_NEAR(q.mean_inlier_error, 0.0, 1e-6);
}

TEST_F(EvaluateTest, DisplacementBeyondTheInlierDistanceScoresZero) {
  PointCloudT shifted;
  for (const auto& p : map_->points) shifted.points.push_back(pt(p.x + 100.0f, p.y, p.z));
  shifted.width = shifted.points.size();
  shifted.height = 1;

  const auto q = evaluate(shifted, tree_, 0.1);
  EXPECT_GT(q.num_evaluated, 0u);
  EXPECT_EQ(q.num_inliers, 0u);
  EXPECT_NEAR(q.inlier_ratio, 0.0, 1e-12);
  EXPECT_FALSE(std::isfinite(q.mean_inlier_error));
}

TEST_F(EvaluateTest, InlierRatioTracksTheOverlapFraction) {
  // Half the source sits on the map, half is 50 m away.
  PointCloudT src;
  for (const auto& p : map_->points) {
    src.points.push_back(p);
    src.points.push_back(pt(p.x + 50.0f, p.y, p.z));
  }
  src.width  = src.points.size();
  src.height = 1;

  const auto q = evaluate(src, tree_, 0.1);
  EXPECT_NEAR(q.inlier_ratio, 0.5, 1e-6);
}

// This is the property the accept/reject gate depends on and that
// pcl::Registration::getFitnessScore() does not have: adding source points that
// fall outside the map changes *how much* matched without changing *how well*
// the matched part fits. A squared-distance average over all points with an
// unbounded search range would move with the outliers instead.
TEST_F(EvaluateTest, MeanInlierErrorIsInvariantToOutOfMapPoints) {
  PointCloudT on_map;
  for (const auto& p : map_->points) on_map.points.push_back(pt(p.x + 0.02f, p.y, p.z));
  on_map.width = on_map.points.size();
  on_map.height = 1;
  const auto baseline = evaluate(on_map, tree_, 0.1);

  PointCloudT with_outliers = on_map;
  for (const auto& p : map_->points) {
    with_outliers.points.push_back(pt(p.x + 500.0f, p.y + 500.0f, p.z));
  }
  with_outliers.width = with_outliers.points.size();
  const auto polluted = evaluate(with_outliers, tree_, 0.1);

  EXPECT_NEAR(polluted.mean_inlier_error, baseline.mean_inlier_error, 1e-6);
  EXPECT_NEAR(polluted.inlier_ratio, 0.5 * baseline.inlier_ratio, 1e-6);
}

TEST_F(EvaluateTest, MaxSamplesBoundsTheWorkAndKeepsTheEstimate) {
  const auto full    = evaluate(*map_, tree_, 0.1);
  const auto sampled = evaluate(*map_, tree_, 0.1, 100);
  EXPECT_LE(sampled.num_evaluated, 101u);
  EXPECT_GT(sampled.num_evaluated, 0u);
  EXPECT_NEAR(sampled.inlier_ratio, full.inlier_ratio, 1e-6);
}

TEST_F(EvaluateTest, NonFinitePointsAreNotCounted) {
  PointCloudT src;
  src.points.push_back(pt(1.0f, 1.0f, 0.0f));
  src.points.push_back(pt(std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.0f));
  src.width  = 2;
  src.height = 1;

  const auto q = evaluate(src, tree_, 0.1);
  EXPECT_EQ(q.num_evaluated, 1u);
  EXPECT_EQ(q.num_inliers, 1u);
}

TEST_F(EvaluateTest, DegenerateInputsReturnZeroRatherThanThrowing) {
  PointCloudT empty;
  EXPECT_EQ(evaluate(empty, tree_, 0.1).num_evaluated, 0u);
  EXPECT_NEAR(evaluate(*map_, tree_, 0.0).inlier_ratio, 0.0, 1e-12);
  EXPECT_NEAR(evaluate(*map_, tree_, -1.0).inlier_ratio, 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// SubmapCropper
// ---------------------------------------------------------------------------

TEST(SubmapCropper, FirstUpdateBuildsAndCropsToTheRadius) {
  auto map = makeGrid(-50.0, 50.0, -50.0, 50.0, 1.0);
  SubmapCropper c(map, /*radius_m=*/10.0, /*refresh_move_m=*/5.0);

  ASSERT_TRUE(c.update(Eigen::Vector3d(0, 0, 0)));
  ASSERT_TRUE(c.valid());
  EXPECT_LT(c.submap()->size(), map->size());
  for (const auto& p : c.submap()->points) {
    EXPECT_LE(Eigen::Vector3d(p.x, p.y, p.z).norm(), 10.0 + 1e-3);
  }
}

TEST(SubmapCropper, DoesNotRebuildUntilTheCentreMovesFarEnough) {
  auto map = makeGrid(-50.0, 50.0, -50.0, 50.0, 1.0);
  SubmapCropper c(map, 10.0, 5.0);

  ASSERT_TRUE(c.update(Eigen::Vector3d(0, 0, 0)));
  EXPECT_EQ(c.rebuildCount(), 1u);

  EXPECT_FALSE(c.update(Eigen::Vector3d(2.0, 0, 0)));   // inside threshold
  EXPECT_FALSE(c.update(Eigen::Vector3d(0, 4.9, 0)));   // still inside
  EXPECT_EQ(c.rebuildCount(), 1u);

  EXPECT_TRUE(c.update(Eigen::Vector3d(6.0, 0, 0)));    // past threshold
  EXPECT_EQ(c.rebuildCount(), 2u);
}

TEST(SubmapCropper, RefreshDistanceIsMeasuredFromTheLastCropNotTheLastQuery) {
  // Creeping forward in sub-threshold steps must eventually trigger a rebuild,
  // otherwise the submap silently falls behind the robot.
  auto map = makeGrid(-50.0, 50.0, -50.0, 50.0, 1.0);
  SubmapCropper c(map, 10.0, 5.0);
  ASSERT_TRUE(c.update(Eigen::Vector3d(0, 0, 0)));

  bool rebuilt = false;
  for (int i = 1; i <= 10; ++i) rebuilt |= c.update(Eigen::Vector3d(i * 1.0, 0, 0));
  EXPECT_TRUE(rebuilt);
  EXPECT_GE(c.rebuildCount(), 2u);
}

TEST(SubmapCropper, ZeroRadiusUsesTheWholeMapExactlyOnce) {
  auto map = makeGrid(-20.0, 20.0, -20.0, 20.0, 1.0);
  SubmapCropper c(map, /*radius_m=*/0.0, /*refresh_move_m=*/1.0);

  ASSERT_TRUE(c.update(Eigen::Vector3d(0, 0, 0)));
  EXPECT_EQ(c.submap()->size(), map->size());

  EXPECT_FALSE(c.update(Eigen::Vector3d(1000.0, 0, 0)));
  EXPECT_EQ(c.rebuildCount(), 1u);
}

TEST(SubmapCropper, EmptyMapIsInvalidAndNeverRebuilds) {
  PointCloudT::Ptr empty(new PointCloudT);
  SubmapCropper c(empty, 10.0, 5.0);
  EXPECT_FALSE(c.update(Eigen::Vector3d(0, 0, 0)));
  EXPECT_FALSE(c.valid());
  EXPECT_EQ(c.rebuildCount(), 0u);
}

TEST(SubmapCropper, NonFiniteCentreIsIgnored) {
  auto map = makeGrid(-20.0, 20.0, -20.0, 20.0, 1.0);
  SubmapCropper c(map, 10.0, 5.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(c.update(Eigen::Vector3d(nan, 0, 0)));
  EXPECT_EQ(c.rebuildCount(), 0u);
}

TEST(SubmapCropper, CentreOutsideTheMapYieldsAnInvalidSubmapNotACrash) {
  // The robot driving off the edge of the prior map must be observable by the
  // caller, not a KD-tree exception on an empty cloud.
  auto map = makeGrid(-20.0, 20.0, -20.0, 20.0, 1.0);
  SubmapCropper c(map, 10.0, 5.0);
  EXPECT_TRUE(c.update(Eigen::Vector3d(1000.0, 1000.0, 0.0)));
  EXPECT_FALSE(c.valid());
  EXPECT_EQ(c.submap()->size(), 0u);
}

// ---------------------------------------------------------------------------
// ScanAccumulator
// ---------------------------------------------------------------------------

namespace {
PointCloudT makeScan(int n = 100) {
  PointCloudT c;
  for (int i = 0; i < n; ++i) c.points.push_back(pt(i * 0.1f, 0.0f, 0.0f));
  c.width  = c.points.size();
  c.height = 1;
  return c;
}
}  // namespace

TEST(ScanAccumulator, StationaryRobotKeepsExactlyOneKeyframe) {
  AccumulatorConfig cfg;
  cfg.min_keyframe_move_m = 0.5;
  ScanAccumulator a(cfg);

  EXPECT_TRUE(a.add(makeScan(), Eigen::Vector3d(0, 0, 0)));
  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(a.add(makeScan(), Eigen::Vector3d(0.01 * (i % 3), 0, 0)));
  }
  EXPECT_EQ(a.size(), 1u);
}

TEST(ScanAccumulator, AcceptsKeyframesOnceTheRobotMoves) {
  AccumulatorConfig cfg;
  cfg.min_keyframe_move_m = 0.5;
  cfg.max_scans           = 100;
  cfg.max_span_m          = 0.0;
  ScanAccumulator a(cfg);

  for (int i = 0; i < 5; ++i) EXPECT_TRUE(a.add(makeScan(), Eigen::Vector3d(i * 1.0, 0, 0)));
  EXPECT_EQ(a.size(), 5u);
  EXPECT_NEAR(a.spanMetres(), 4.0, 1e-9);
}

TEST(ScanAccumulator, EvictsOldestBeyondMaxScans) {
  AccumulatorConfig cfg;
  cfg.max_scans           = 3;
  cfg.max_span_m          = 0.0;
  cfg.min_keyframe_move_m = 0.5;
  ScanAccumulator a(cfg);

  for (int i = 0; i < 10; ++i) a.add(makeScan(), Eigen::Vector3d(i * 1.0, 0, 0));
  EXPECT_EQ(a.size(), 3u);
  EXPECT_NEAR(a.spanMetres(), 2.0, 1e-9);
}

TEST(ScanAccumulator, EvictsOnPathLengthSpan) {
  AccumulatorConfig cfg;
  cfg.max_scans           = 1000;
  cfg.max_span_m          = 5.0;
  cfg.min_keyframe_move_m = 0.5;
  ScanAccumulator a(cfg);

  for (int i = 0; i < 20; ++i) a.add(makeScan(), Eigen::Vector3d(i * 1.0, 0, 0));
  EXPECT_LE(a.spanMetres(), 5.0 + 1e-9);
  EXPECT_GT(a.size(), 1u);
}

TEST(ScanAccumulator, SpanEvictionNeverEmptiesTheWindow) {
  // A single step longer than max_span must still leave the newest keyframe.
  AccumulatorConfig cfg;
  cfg.max_span_m          = 1.0;
  cfg.min_keyframe_move_m = 0.0;
  ScanAccumulator a(cfg);

  a.add(makeScan(), Eigen::Vector3d(0, 0, 0));
  a.add(makeScan(), Eigen::Vector3d(100.0, 0, 0));
  EXPECT_EQ(a.size(), 1u);
  EXPECT_FALSE(a.empty());
}

TEST(ScanAccumulator, RejectsEmptyScansAndNonFinitePositions) {
  ScanAccumulator a;
  PointCloudT empty;
  EXPECT_FALSE(a.add(empty, Eigen::Vector3d(0, 0, 0)));

  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(a.add(makeScan(), Eigen::Vector3d(nan, 0, 0)));
  EXPECT_EQ(a.size(), 0u);
}

TEST(ScanAccumulator, FiltersNonFinitePointsAndRejectsAnAllNaNScan) {
  ScanAccumulator a;
  PointCloudT mixed;
  mixed.points.push_back(pt(1.0f, 1.0f, 1.0f));
  mixed.points.push_back(pt(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f));
  mixed.width = 2;
  mixed.height = 1;
  EXPECT_TRUE(a.add(mixed, Eigen::Vector3d(0, 0, 0)));
  EXPECT_EQ(a.merged()->size(), 1u);

  ScanAccumulator b;
  PointCloudT all_nan;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  all_nan.points.push_back(pt(nan, nan, nan));
  all_nan.width = 1;
  all_nan.height = 1;
  EXPECT_FALSE(b.add(all_nan, Eigen::Vector3d(0, 0, 0)));
}

TEST(ScanAccumulator, MergedIsNonNullWhenEmpty) {
  ScanAccumulator a;
  auto m = a.merged();
  ASSERT_NE(m, nullptr);
  EXPECT_TRUE(m->empty());
}

TEST(ScanAccumulator, MergedAppliesTheOutputVoxel) {
  AccumulatorConfig cfg;
  cfg.min_keyframe_move_m = 0.0;
  cfg.voxel_size          = 1.0;
  ScanAccumulator a(cfg);
  // 100 points spaced 0.1 m along x -> ~10 occupied 1 m voxels.
  a.add(makeScan(100), Eigen::Vector3d(0, 0, 0));
  EXPECT_LT(a.merged()->size(), 20u);
  EXPECT_GT(a.merged()->size(), 0u);
}

TEST(ScanAccumulator, ClearResetsTheKeyframeGate) {
  AccumulatorConfig cfg;
  cfg.min_keyframe_move_m = 0.5;
  ScanAccumulator a(cfg);
  a.add(makeScan(), Eigen::Vector3d(0, 0, 0));
  a.clear();
  EXPECT_EQ(a.size(), 0u);
  // Same position as before must be accepted again after a clear.
  EXPECT_TRUE(a.add(makeScan(), Eigen::Vector3d(0, 0, 0)));
}

TEST(ScanAccumulator, ZeroMaxScansIsCoercedToOne) {
  AccumulatorConfig cfg;
  cfg.max_scans           = 0;
  cfg.min_keyframe_move_m = 0.0;
  ScanAccumulator a(cfg);
  a.add(makeScan(), Eigen::Vector3d(0, 0, 0));
  a.add(makeScan(), Eigen::Vector3d(1, 0, 0));
  EXPECT_EQ(a.size(), 1u);
}

// ---------------------------------------------------------------------------
// applyCorrection
// ---------------------------------------------------------------------------

namespace {
RegistrationQuality goodQuality(double ratio = 0.9) {
  RegistrationQuality q;
  q.inlier_ratio      = ratio;
  q.mean_inlier_error = 0.05;
  q.num_inliers       = 900;
  q.num_evaluated     = 1000;
  return q;
}
}  // namespace

TEST(ApplyCorrection, AcceptsAndMovesPartwayTowardTheCandidate) {
  CorrectionPolicy p;
  p.alpha                  = 0.5;
  p.max_translation_step_m = 1.0;

  const auto current   = makePose(0, 0, 0, 0.0);
  const auto candidate = makePose(0.4, 0, 0, 0.0);

  const auto r = applyCorrection(current, candidate, goodQuality(), p);
  ASSERT_TRUE(r.accepted());
  EXPECT_NEAR(r.pose.translation().x(), 0.2, 1e-9);
  EXPECT_NEAR(r.delta.translation_m, 0.4, 1e-9);
}

TEST(ApplyCorrection, AlphaOfOneReproducesTheOldHardOverwrite) {
  CorrectionPolicy p;
  p.alpha = 1.0;
  const auto current   = makePose(0, 0, 0, 0.0);
  const auto candidate = makePose(0.4, 0.1, 0, 2.0);
  const auto r = applyCorrection(current, candidate, goodQuality(), p);
  ASSERT_TRUE(r.accepted());
  EXPECT_NEAR(poseDelta(r.pose, candidate).translation_m, 0.0, 1e-9);
  EXPECT_NEAR(poseDelta(r.pose, candidate).rotation_rad, 0.0, 1e-7);
}

TEST(ApplyCorrection, AlphaOfZeroAcceptsButDoesNotMove) {
  CorrectionPolicy p;
  p.alpha = 0.0;
  const auto current   = makePose(0, 0, 0, 0.0);
  const auto candidate = makePose(0.4, 0, 0, 0.0);
  const auto r = applyCorrection(current, candidate, goodQuality(), p);
  EXPECT_TRUE(r.accepted());
  EXPECT_NEAR(poseDelta(r.pose, current).translation_m, 0.0, 1e-9);
}

TEST(ApplyCorrection, RejectsLowQualityAndLeavesThePoseUntouched) {
  CorrectionPolicy p;
  p.min_inlier_ratio = 0.5;

  const auto current   = makePose(1, 2, 3, 10.0);
  const auto candidate = makePose(1.1, 2, 3, 10.0);

  const auto r = applyCorrection(current, candidate, goodQuality(0.2), p);
  EXPECT_EQ(r.verdict, CorrectionVerdict::kRejectedQuality);
  EXPECT_NEAR(poseDelta(r.pose, current).translation_m, 0.0, 1e-12);
}

TEST(ApplyCorrection, RejectsAnImplausibleTranslationJump) {
  CorrectionPolicy p;
  p.max_translation_step_m = 0.5;

  const auto current   = makePose(0, 0, 0, 0.0);
  const auto candidate = makePose(30.0, 0, 0, 0.0);  // aliased onto the wrong aisle

  const auto r = applyCorrection(current, candidate, goodQuality(0.95), p);
  EXPECT_EQ(r.verdict, CorrectionVerdict::kRejectedJump);
  EXPECT_NEAR(poseDelta(r.pose, current).translation_m, 0.0, 1e-12);
}

TEST(ApplyCorrection, RejectsAnImplausibleRotationJumpEvenWithGoodQuality) {
  CorrectionPolicy p;
  p.max_translation_step_m = 10.0;
  p.max_rotation_step_rad  = 5.0 * kDeg;

  const auto current   = makePose(0, 0, 0, 0.0);
  const auto candidate = makePose(0, 0, 0, 180.0);  // corridor flipped end for end

  const auto r = applyCorrection(current, candidate, goodQuality(0.99), p);
  EXPECT_EQ(r.verdict, CorrectionVerdict::kRejectedJump);
}

TEST(ApplyCorrection, RejectsNonFiniteCandidates) {
  CorrectionPolicy p;
  auto candidate = makePose(0.1, 0, 0, 0.0);
  candidate.translation().z() = std::numeric_limits<double>::quiet_NaN();

  const auto current = makePose(0, 0, 0, 0.0);
  const auto r = applyCorrection(current, candidate, goodQuality(), p);
  EXPECT_EQ(r.verdict, CorrectionVerdict::kRejectedInvalid);
  EXPECT_TRUE(isFinite(r.pose));
}

TEST(ApplyCorrection, NonPositiveStepLimitsDisableTheJumpGate) {
  CorrectionPolicy p;
  p.alpha                  = 1.0;
  p.max_translation_step_m = 0.0;
  p.max_rotation_step_rad  = 0.0;

  const auto current   = makePose(0, 0, 0, 0.0);
  const auto candidate = makePose(500.0, 0, 0, 179.0);
  EXPECT_TRUE(applyCorrection(current, candidate, goodQuality(), p).accepted());
}

TEST(ApplyCorrection, RepeatedAcceptedCorrectionsConvergeAndStayOrthonormal) {
  // The interpolated update is applied in a loop on the robot; it must not
  // accumulate scale drift or stall short of the target.
  CorrectionPolicy p;
  p.alpha                  = 0.3;
  p.max_translation_step_m = 5.0;
  p.max_rotation_step_rad  = M_PI;

  const auto target = makePose(2.0, -1.0, 0.3, 25.0);
  auto current      = makePose(0, 0, 0, 0.0);
  for (int i = 0; i < 100; ++i) {
    const auto r = applyCorrection(current, target, goodQuality(), p);
    ASSERT_TRUE(r.accepted()) << "iteration " << i;
    current = r.pose;
  }
  EXPECT_NEAR(poseDelta(current, target).translation_m, 0.0, 1e-6);
  EXPECT_NEAR(poseDelta(current, target).rotation_rad, 0.0, 1e-6);
  EXPECT_NEAR(current.linear().determinant(), 1.0, 1e-9);
}
