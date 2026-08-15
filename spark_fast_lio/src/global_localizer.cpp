#include "global_localizer.h"

#include <chrono>
#include <utility>

#ifdef SPARK_FAST_LIO_HAS_KISS_MATCHER
#include <kiss_matcher/KISSMatcher.hpp>
#endif

namespace spark_fast_lio {
namespace reloc {

GlobalLocalizer::GlobalLocalizer(GlobalLocalizerConfig cfg,
                                 std::shared_ptr<GlobalRegistrationBackend> backend)
    : cfg_(cfg), backend_(std::move(backend)) {
  if (cfg_.required_confirmations < 1) cfg_.required_confirmations = 1;
}

Proposal GlobalLocalizer::propose(const PointCloudT& source_in_odom,
                                  const std::vector<Eigen::Vector3f>& target_map) {
  Proposal p;
  if (!backend_) {
    p.status = ProposalStatus::kBackendFailed;
    return p;
  }

  const std::vector<Eigen::Vector3f> source = toVector(source_in_odom);
  if (source.size() < cfg_.min_source_points || target_map.empty()) {
    p.status = ProposalStatus::kInsufficientInput;
    return p;
  }

  ++attempt_count_;
  p.registration = backend_->align(source, target_map);

  if (!p.registration.valid || !isFinite(p.registration.transform)) {
    p.status = ProposalStatus::kBackendFailed;
    return p;
  }
  if (p.registration.num_final_inliers < cfg_.min_final_inliers) {
    p.status = ProposalStatus::kTooFewInliers;
    return p;
  }

  p.status     = ProposalStatus::kOk;
  p.map_T_odom = orthonormalized(p.registration.transform);
  return p;
}

Verification GlobalLocalizer::submitVerification(const Eigen::Isometry3d& refined_map_T_odom,
                                                 const RegistrationQuality& quality) {
  Verification v;
  v.map_T_odom = refined_map_T_odom;

  if (!isFinite(refined_map_T_odom)) {
    v.status        = VerificationStatus::kRejectedInvalid;
    v.confirmations = confirmations_;
    return v;
  }

  if (quality.inlier_ratio < cfg_.min_verify_inlier_ratio) {
    // A failed verification is evidence against the streak, not neutral: keep
    // it from carrying over into the next proposal.
    confirmations_    = 0;
    has_corroborated_ = false;
    v.status          = VerificationStatus::kRejectedQuality;
    v.confirmations   = 0;
    return v;
  }

  if (has_corroborated_) {
    v.disagreement = poseDelta(refined_map_T_odom, last_corroborated_);
    const bool agrees = v.disagreement.translation_m <= cfg_.max_confirm_disagreement_m &&
                        v.disagreement.rotation_rad <= cfg_.max_confirm_disagreement_rad;
    confirmations_ = agrees ? confirmations_ + 1 : 1;
  } else {
    confirmations_ = 1;
  }

  last_corroborated_ = refined_map_T_odom;
  has_corroborated_  = true;
  v.confirmations    = confirmations_;
  v.status = (confirmations_ >= cfg_.required_confirmations) ? VerificationStatus::kConfirmed
                                                             : VerificationStatus::kPending;
  return v;
}

void GlobalLocalizer::reset() {
  confirmations_     = 0;
  has_corroborated_  = false;
  last_corroborated_ = Eigen::Isometry3d::Identity();
}

const char* toString(ProposalStatus s) {
  switch (s) {
    case ProposalStatus::kOk:                return "ok";
    case ProposalStatus::kInsufficientInput: return "insufficient-input";
    case ProposalStatus::kBackendFailed:     return "backend-failed";
    case ProposalStatus::kTooFewInliers:     return "too-few-inliers";
  }
  return "unknown";
}

const char* toString(VerificationStatus s) {
  switch (s) {
    case VerificationStatus::kRejectedQuality: return "rejected:quality";
    case VerificationStatus::kRejectedInvalid: return "rejected:invalid";
    case VerificationStatus::kPending:         return "pending";
    case VerificationStatus::kConfirmed:       return "confirmed";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// KISS-Matcher backend
// ---------------------------------------------------------------------------

#ifdef SPARK_FAST_LIO_HAS_KISS_MATCHER

KissMatcherBackend::KissMatcherBackend(const GlobalLocalizerConfig& cfg) : cfg_(cfg) {}

GlobalRegistrationResult KissMatcherBackend::align(const std::vector<Eigen::Vector3f>& source,
                                                   const std::vector<Eigen::Vector3f>& target) {
  GlobalRegistrationResult out;

  // KISSMatcherConfig's constructor throws on a voxel size below 5 mm and on an
  // inconsistent noise-bound pair, so a bad YAML value must not take the node
  // down mid-mission.
  std::unique_ptr<kiss_matcher::KISSMatcher> matcher;
  try {
    kiss_matcher::KISSMatcherConfig kcfg(cfg_.voxel_size,
                                         /*use_voxel_sampling=*/true,
                                         /*use_quatro=*/cfg_.use_quatro);
    kcfg.use_ratio_test_ = cfg_.use_ratio_test;
    matcher              = std::make_unique<kiss_matcher::KISSMatcher>(kcfg);
  } catch (const std::exception&) {
    out.valid = false;
    return out;
  }

  const auto t0 = std::chrono::steady_clock::now();
  kiss_matcher::RegistrationSolution solution;
  try {
    solution = matcher->estimate(source, target);
  } catch (const std::exception&) {
    out.valid = false;
    return out;
  }
  const auto t1 = std::chrono::steady_clock::now();

  out.seconds = std::chrono::duration<double>(t1 - t0).count();
  out.num_rotation_inliers = matcher->getNumRotationInliers();
  out.num_final_inliers    = matcher->getNumFinalInliers();

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear()          = solution.rotation;
  T.translation()     = solution.translation;

  // `solution.valid` is only part of the story; a degenerate solve can still
  // come back flagged valid with a non-finite or non-rotation matrix.
  out.transform = T;
  out.valid     = solution.valid && T.matrix().allFinite();
  return out;
}

#endif  // SPARK_FAST_LIO_HAS_KISS_MATCHER

}  // namespace reloc
}  // namespace spark_fast_lio
