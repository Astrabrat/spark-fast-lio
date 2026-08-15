// Tests for the global relocalization policy.
//
// The backend is faked so these exercise the decision logic rather than
// KISS-Matcher itself: what gets rejected, what gets latched, and — most
// importantly — that a confident but wrong registration cannot latch on its
// own. That last case is the one that silently ruins a mission, because a
// self-similar environment produces geometrically excellent fits to the wrong
// part of the map.

#include "global_localizer.h"

#include <cmath>
#include <deque>

#include <gtest/gtest.h>

using namespace spark_fast_lio::reloc;

namespace {

constexpr double kDeg = M_PI / 180.0;

Eigen::Isometry3d makePose(double x, double y, double z, double yaw_deg) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(yaw_deg * kDeg, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T.translation() = Eigen::Vector3d(x, y, z);
  return T;
}

// Returns pre-programmed results in order; repeats the last one once exhausted.
class FakeBackend : public GlobalRegistrationBackend {
 public:
  void push(GlobalRegistrationResult r) { queued_.push_back(r); }

  GlobalRegistrationResult align(const std::vector<Eigen::Vector3f>& source,
                                 const std::vector<Eigen::Vector3f>& target) override {
    ++calls;
    last_source_size = source.size();
    last_target_size = target.size();
    if (queued_.empty()) return last_;
    last_ = queued_.front();
    if (queued_.size() > 1) queued_.pop_front();
    return last_;
  }
  std::string name() const override { return "fake"; }

  int calls{0};
  std::size_t last_source_size{0};
  std::size_t last_target_size{0};

 private:
  std::deque<GlobalRegistrationResult> queued_;
  GlobalRegistrationResult last_;
};

GlobalRegistrationResult goodResult(const Eigen::Isometry3d& T, std::size_t inliers = 200) {
  GlobalRegistrationResult r;
  r.valid                = true;
  r.transform            = T;
  r.num_final_inliers    = inliers;
  r.num_rotation_inliers = inliers + 10;
  r.seconds              = 0.5;
  return r;
}

RegistrationQuality quality(double ratio) {
  RegistrationQuality q;
  q.inlier_ratio      = ratio;
  q.mean_inlier_error = 0.08;
  q.num_evaluated     = 1000;
  q.num_inliers       = static_cast<std::size_t>(ratio * 1000);
  return q;
}

PointCloudT makeSource(std::size_t n) {
  PointCloudT c;
  for (std::size_t i = 0; i < n; ++i) {
    PointT p;
    p.x = static_cast<float>(i % 100) * 0.1f;
    p.y = static_cast<float>((i / 100) % 100) * 0.1f;
    p.z = 0.0f;
    p.intensity = 0.0f;
    c.points.push_back(p);
  }
  c.width  = static_cast<uint32_t>(c.points.size());
  c.height = 1;
  return c;
}

std::vector<Eigen::Vector3f> makeTarget(std::size_t n) {
  std::vector<Eigen::Vector3f> v;
  for (std::size_t i = 0; i < n; ++i) v.emplace_back(static_cast<float>(i), 0.0f, 0.0f);
  return v;
}

GlobalLocalizerConfig defaultCfg() {
  GlobalLocalizerConfig c;
  c.min_source_points            = 100;
  c.min_final_inliers            = 20;
  c.min_verify_inlier_ratio      = 0.5;
  c.required_confirmations       = 2;
  c.max_confirm_disagreement_m   = 1.0;
  c.max_confirm_disagreement_rad = 0.1;
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// propose
// ---------------------------------------------------------------------------

TEST(Propose, RejectsATooSmallSourceWithoutCallingTheBackend) {
  auto backend = std::make_shared<FakeBackend>();
  backend->push(goodResult(makePose(10, 0, 0, 0)));
  GlobalLocalizer g(defaultCfg(), backend);

  const auto p = g.propose(makeSource(50), makeTarget(1000));
  EXPECT_EQ(p.status, ProposalStatus::kInsufficientInput);
  EXPECT_EQ(backend->calls, 0) << "a too-small source must not pay for a global solve";
  EXPECT_EQ(g.attemptCount(), 0u);
}

TEST(Propose, RejectsAnEmptyTarget) {
  auto backend = std::make_shared<FakeBackend>();
  GlobalLocalizer g(defaultCfg(), backend);
  const auto p = g.propose(makeSource(500), {});
  EXPECT_EQ(p.status, ProposalStatus::kInsufficientInput);
  EXPECT_EQ(backend->calls, 0);
}

TEST(Propose, ReportsBackendFailure) {
  auto backend = std::make_shared<FakeBackend>();
  GlobalRegistrationResult bad;
  bad.valid = false;
  backend->push(bad);
  GlobalLocalizer g(defaultCfg(), backend);

  EXPECT_EQ(g.propose(makeSource(500), makeTarget(1000)).status, ProposalStatus::kBackendFailed);
}

TEST(Propose, RejectsANonFiniteTransformEvenWhenTheBackendSaysValid) {
  auto backend = std::make_shared<FakeBackend>();
  auto r = goodResult(makePose(1, 2, 3, 10));
  r.transform.translation().x() = std::numeric_limits<double>::quiet_NaN();
  backend->push(r);
  GlobalLocalizer g(defaultCfg(), backend);

  EXPECT_EQ(g.propose(makeSource(500), makeTarget(1000)).status, ProposalStatus::kBackendFailed);
}

TEST(Propose, RejectsTooFewFinalInliers) {
  auto backend = std::make_shared<FakeBackend>();
  backend->push(goodResult(makePose(10, 0, 0, 0), /*inliers=*/5));
  GlobalLocalizer g(defaultCfg(), backend);

  const auto p = g.propose(makeSource(500), makeTarget(1000));
  EXPECT_EQ(p.status, ProposalStatus::kTooFewInliers);
  EXPECT_EQ(p.registration.num_final_inliers, 5u);
}

TEST(Propose, AcceptsAGoodResultAndReturnsAProperRotation) {
  auto backend = std::make_shared<FakeBackend>();
  auto r = goodResult(makePose(12.0, -3.0, 0.5, 40.0));
  r.transform.linear() *= 1.02;  // GNC output that drifted off SO(3)
  backend->push(r);
  GlobalLocalizer g(defaultCfg(), backend);

  const auto p = g.propose(makeSource(500), makeTarget(1000));
  ASSERT_TRUE(p.ok());
  EXPECT_NEAR(p.map_T_odom.linear().determinant(), 1.0, 1e-9);
  EXPECT_TRUE(p.map_T_odom.translation().isApprox(Eigen::Vector3d(12.0, -3.0, 0.5), 1e-9));
  EXPECT_EQ(g.attemptCount(), 1u);
}

TEST(Propose, DropsNonFiniteSourcePointsBeforeSizeGating) {
  auto backend = std::make_shared<FakeBackend>();
  backend->push(goodResult(makePose(0, 0, 0, 0)));
  GlobalLocalizer g(defaultCfg(), backend);

  PointCloudT src = makeSource(500);
  for (int i = 0; i < 450; ++i) src.points[i].x = std::numeric_limits<float>::quiet_NaN();

  // 50 finite points is below min_source_points(100), so this must not reach
  // the backend just because the raw cloud looked big enough.
  EXPECT_EQ(g.propose(src, makeTarget(1000)).status, ProposalStatus::kInsufficientInput);
  EXPECT_EQ(backend->calls, 0);
}

TEST(Propose, ANullBackendFailsClosed) {
  GlobalLocalizer g(defaultCfg(), nullptr);
  EXPECT_EQ(g.propose(makeSource(500), makeTarget(1000)).status, ProposalStatus::kBackendFailed);
}

// ---------------------------------------------------------------------------
// submitVerification
// ---------------------------------------------------------------------------

class VerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = std::make_shared<FakeBackend>();
    g_ = std::make_unique<GlobalLocalizer>(defaultCfg(), backend_);
  }
  std::shared_ptr<FakeBackend> backend_;
  std::unique_ptr<GlobalLocalizer> g_;
};

TEST_F(VerifyTest, OneGoodProposalIsPendingNotConfirmed) {
  const auto v = g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9));
  EXPECT_EQ(v.status, VerificationStatus::kPending);
  EXPECT_EQ(v.confirmations, 1);
}

TEST_F(VerifyTest, TwoAgreeingProposalsConfirm) {
  ASSERT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kPending);
  const auto v = g_->submitVerification(makePose(10.2, 0.1, 0, 1.0), quality(0.88));
  EXPECT_EQ(v.status, VerificationStatus::kConfirmed);
  EXPECT_EQ(v.confirmations, 2);
  EXPECT_LT(v.disagreement.translation_m, 1.0);
}

// The core anti-aliasing property: two confident, high-quality registrations
// that place the robot in different places must never latch.
TEST_F(VerifyTest, DisagreeingProposalsNeverConfirmNoMatterHowConfident) {
  for (int i = 0; i < 20; ++i) {
    const auto pose = (i % 2 == 0) ? makePose(10, 0, 0, 0) : makePose(40, 0, 0, 0);
    const auto v    = g_->submitVerification(pose, quality(0.99));
    EXPECT_NE(v.status, VerificationStatus::kConfirmed) << "iteration " << i;
    EXPECT_EQ(v.confirmations, 1);
  }
}

TEST_F(VerifyTest, DisagreementResetsTheStreakToOneNotZero) {
  // The disagreeing proposal is itself corroborated, so it becomes the new
  // hypothesis rather than being discarded outright.
  ASSERT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9)).confirmations, 1);
  const auto v = g_->submitVerification(makePose(40, 0, 0, 0), quality(0.9));
  EXPECT_EQ(v.confirmations, 1);
  EXPECT_EQ(v.status, VerificationStatus::kPending);
  EXPECT_NEAR(v.disagreement.translation_m, 30.0, 1e-6);

  // ... and the new hypothesis can then confirm on its own.
  EXPECT_EQ(g_->submitVerification(makePose(40.1, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kConfirmed);
}

TEST_F(VerifyTest, YawDisagreementAloneBlocksConfirmation) {
  // Same position, opposite heading: the classic corridor 180 flip.
  ASSERT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.95)).confirmations, 1);
  const auto v = g_->submitVerification(makePose(10, 0, 0, 180.0), quality(0.95));
  EXPECT_EQ(v.status, VerificationStatus::kPending);
  EXPECT_EQ(v.confirmations, 1);
}

TEST_F(VerifyTest, LowQualityIsRejectedAndBreaksTheStreak) {
  ASSERT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9)).confirmations, 1);

  const auto bad = g_->submitVerification(makePose(10, 0, 0, 0), quality(0.1));
  EXPECT_EQ(bad.status, VerificationStatus::kRejectedQuality);
  EXPECT_EQ(bad.confirmations, 0);
  EXPECT_EQ(g_->confirmations(), 0);

  // A single good proposal after the break must not immediately confirm.
  EXPECT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kPending);
}

TEST_F(VerifyTest, NonFinitePoseIsRejectedWithoutDisturbingTheStreak) {
  ASSERT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9)).confirmations, 1);

  auto nan_pose = makePose(10, 0, 0, 0);
  nan_pose.translation().z() = std::numeric_limits<double>::quiet_NaN();
  const auto v = g_->submitVerification(nan_pose, quality(0.9));
  EXPECT_EQ(v.status, VerificationStatus::kRejectedInvalid);
  EXPECT_EQ(g_->confirmations(), 1);
}

TEST_F(VerifyTest, ResetClearsTheStreak) {
  ASSERT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9)).confirmations, 1);
  g_->reset();
  EXPECT_EQ(g_->confirmations(), 0);
  EXPECT_EQ(g_->submitVerification(makePose(10, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kPending);
}

TEST(VerifyConfig, RequiredConfirmationsOfOneLatchesImmediately) {
  auto cfg = defaultCfg();
  cfg.required_confirmations = 1;
  GlobalLocalizer g(cfg, std::make_shared<FakeBackend>());
  EXPECT_EQ(g.submitVerification(makePose(10, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kConfirmed);
}

TEST(VerifyConfig, ZeroOrNegativeRequiredConfirmationsIsCoercedToOne) {
  auto cfg = defaultCfg();
  cfg.required_confirmations = 0;
  GlobalLocalizer g(cfg, std::make_shared<FakeBackend>());
  EXPECT_EQ(g.config().required_confirmations, 1);
  EXPECT_EQ(g.submitVerification(makePose(0, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kConfirmed);
}

TEST(VerifyConfig, ThreeConfirmationsRequiresThreeAgreeingProposals) {
  auto cfg = defaultCfg();
  cfg.required_confirmations = 3;
  GlobalLocalizer g(cfg, std::make_shared<FakeBackend>());

  EXPECT_EQ(g.submitVerification(makePose(5, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kPending);
  EXPECT_EQ(g.submitVerification(makePose(5.1, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kPending);
  EXPECT_EQ(g.submitVerification(makePose(5.2, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kConfirmed);
}

TEST(VerifyConfig, AgreementIsCheckedAgainstThePreviousProposalNotTheFirst) {
  // Documents the drift caveat: agreement is pairwise, so a slow walk of
  // sub-tolerance steps can confirm at a pose further from the first proposal
  // than the tolerance. The step limit in applyCorrection is what bounds the
  // consequences downstream.
  auto cfg = defaultCfg();
  cfg.required_confirmations     = 3;
  cfg.max_confirm_disagreement_m = 1.0;
  GlobalLocalizer g(cfg, std::make_shared<FakeBackend>());

  EXPECT_EQ(g.submitVerification(makePose(0, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kPending);
  EXPECT_EQ(g.submitVerification(makePose(0.9, 0, 0, 0), quality(0.9)).status,
            VerificationStatus::kPending);
  const auto v = g.submitVerification(makePose(1.8, 0, 0, 0), quality(0.9));
  EXPECT_EQ(v.status, VerificationStatus::kConfirmed);
  EXPECT_NEAR(v.map_T_odom.translation().x(), 1.8, 1e-9);
}

TEST(VerifyConfig, QualityExactlyAtTheThresholdIsAccepted) {
  auto cfg = defaultCfg();
  cfg.min_verify_inlier_ratio = 0.5;
  GlobalLocalizer g(cfg, std::make_shared<FakeBackend>());
  EXPECT_NE(g.submitVerification(makePose(0, 0, 0, 0), quality(0.5)).status,
            VerificationStatus::kRejectedQuality);
}

// ---------------------------------------------------------------------------
// End-to-end policy walk
// ---------------------------------------------------------------------------

TEST(Policy, TypicalStartupSequenceLatchesOnTheSecondGoodProposal) {
  auto backend = std::make_shared<FakeBackend>();
  const auto truth = makePose(37.5, -12.0, 0.2, 115.0);
  backend->push(goodResult(makePose(37.4, -12.1, 0.2, 114.5)));
  backend->push(goodResult(truth));

  GlobalLocalizer g(defaultCfg(), backend);

  const auto p1 = g.propose(makeSource(5000), makeTarget(50000));
  ASSERT_TRUE(p1.ok());
  EXPECT_EQ(g.submitVerification(p1.map_T_odom, quality(0.72)).status,
            VerificationStatus::kPending);

  const auto p2 = g.propose(makeSource(5000), makeTarget(50000));
  ASSERT_TRUE(p2.ok());
  const auto v2 = g.submitVerification(p2.map_T_odom, quality(0.81));
  ASSERT_EQ(v2.status, VerificationStatus::kConfirmed);
  EXPECT_NEAR(poseDelta(v2.map_T_odom, truth).translation_m, 0.0, 1e-6);
  EXPECT_EQ(backend->calls, 2);
}

TEST(Policy, AFailedProposalDoesNotConsumeAConfirmation) {
  auto backend = std::make_shared<FakeBackend>();
  backend->push(goodResult(makePose(10, 0, 0, 0)));
  backend->push(goodResult(makePose(10, 0, 0, 0), /*inliers=*/2));  // fails the inlier gate
  backend->push(goodResult(makePose(10.1, 0, 0, 0)));

  GlobalLocalizer g(defaultCfg(), backend);

  const auto p1 = g.propose(makeSource(5000), makeTarget(50000));
  ASSERT_TRUE(p1.ok());
  ASSERT_EQ(g.submitVerification(p1.map_T_odom, quality(0.9)).confirmations, 1);

  const auto p2 = g.propose(makeSource(5000), makeTarget(50000));
  ASSERT_EQ(p2.status, ProposalStatus::kTooFewInliers);
  EXPECT_EQ(g.confirmations(), 1) << "a rejected proposal is not evidence either way";

  const auto p3 = g.propose(makeSource(5000), makeTarget(50000));
  ASSERT_TRUE(p3.ok());
  EXPECT_EQ(g.submitVerification(p3.map_T_odom, quality(0.9)).status,
            VerificationStatus::kConfirmed);
}
