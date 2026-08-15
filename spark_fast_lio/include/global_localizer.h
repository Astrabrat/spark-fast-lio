// Global relocalization against a prior map, with no operator-supplied initial
// pose.
//
// Stage 1 (`propose`) runs a correspondence-based global registration backend —
// KISS-Matcher in production — which needs no initial guess. Stage 2
// (`submitVerification`) folds in a local refinement scored by the caller and
// requires several consecutive proposals to agree before latching.
//
// The two-stage split exists because a single global registration is not
// trustworthy on its own: in a self-similar environment the solver can return a
// geometrically excellent fit to the *wrong* part of the map, with a healthy
// inlier count. Requiring agreement across proposals taken from different
// viewpoints is what separates a true match from an aliased one.
//
// The backend is injected so this policy is testable without KISS-Matcher
// linked in.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "reloc_core.h"

namespace spark_fast_lio {
namespace reloc {

struct GlobalRegistrationResult {
  bool valid{false};
  // Maps points from the source frame into the target frame.
  Eigen::Isometry3d transform{Eigen::Isometry3d::Identity()};
  std::size_t num_rotation_inliers{0};
  // The GNC/COTE translation inlier count. KISS-Matcher's own guidance is that
  // this is the number to threshold on to decide success.
  std::size_t num_final_inliers{0};
  double seconds{0.0};
};

class GlobalRegistrationBackend {
 public:
  virtual ~GlobalRegistrationBackend() = default;
  virtual GlobalRegistrationResult align(const std::vector<Eigen::Vector3f>& source,
                                         const std::vector<Eigen::Vector3f>& target) = 0;
  virtual std::string name() const = 0;
};

struct GlobalLocalizerConfig {
  // KISS-Matcher voxel size. Everything else in KISSMatcherConfig is derived
  // from it (normal radius 3x, FPFH radius 5x, noise bounds 1.0x/0.75x), so it
  // is the one knob that matters. Larger is faster and more robust at map
  // scale; smaller is more precise but slower and more outlier prone.
  float voxel_size{0.5f};
  // Ground robots rotate essentially about yaw only, where Quatro's decoupled
  // estimation is better conditioned than full SO(3) GNC.
  bool use_quatro{true};
  bool use_ratio_test{true};

  // Reject a proposal below this many final inliers. KISS-Matcher's example
  // uses 5 as a bare-minimum sanity check; at map scale a much higher bar is
  // appropriate because false positives are the failure mode that matters.
  std::size_t min_final_inliers{20};
  // Too few source points and FPFH has nothing discriminative to key on.
  std::size_t min_source_points{2000};

  // Verification (stage 2), scored by the caller against the prior map.
  double min_verify_inlier_ratio{0.5};

  // Consecutive agreeing, verified proposals required before latching.
  int required_confirmations{2};
  // Two proposals "agree" if they are within both tolerances.
  double max_confirm_disagreement_m{1.0};
  double max_confirm_disagreement_rad{0.1};
};

enum class ProposalStatus {
  kOk,
  kInsufficientInput,   // source cloud too small or empty target
  kBackendFailed,       // solver returned invalid / non-finite
  kTooFewInliers,
};

struct Proposal {
  ProposalStatus status{ProposalStatus::kInsufficientInput};
  Eigen::Isometry3d map_T_odom{Eigen::Isometry3d::Identity()};
  GlobalRegistrationResult registration;
  bool ok() const { return status == ProposalStatus::kOk; }
};

enum class VerificationStatus {
  kRejectedQuality,   // local refinement did not corroborate the proposal
  kRejectedInvalid,   // non-finite pose
  kPending,           // corroborated, but not yet enough agreeing proposals
  kConfirmed,         // latched: safe to start tracking from this pose
};

struct Verification {
  VerificationStatus status{VerificationStatus::kRejectedQuality};
  Eigen::Isometry3d map_T_odom{Eigen::Isometry3d::Identity()};
  int confirmations{0};
  // Disagreement against the previously corroborated proposal. Zero on the
  // first corroborated proposal.
  PoseDelta disagreement;
  bool confirmed() const { return status == VerificationStatus::kConfirmed; }
};

class GlobalLocalizer {
 public:
  GlobalLocalizer(GlobalLocalizerConfig cfg,
                  std::shared_ptr<GlobalRegistrationBackend> backend);

  // `source_in_odom` is the accumulated local submap in the odom frame;
  // `target_map` is the prior map in the map frame. The returned transform is
  // therefore map_T_odom directly — no composition with the LIO pose is needed,
  // which is what makes this robust to the source cloud being stale by the time
  // the (slow) global registration finishes.
  Proposal propose(const PointCloudT& source_in_odom,
                   const std::vector<Eigen::Vector3f>& target_map);

  // `refined_map_T_odom` is the proposal after local refinement (GICP), and
  // `quality` is that refinement scored against the prior map.
  Verification submitVerification(const Eigen::Isometry3d& refined_map_T_odom,
                                  const RegistrationQuality& quality);

  // Drops the confirmation streak. Call on relocalization loss so a stale
  // partial streak cannot combine with a fresh proposal to latch early.
  void reset();

  int confirmations() const { return confirmations_; }
  const GlobalLocalizerConfig& config() const { return cfg_; }
  std::size_t attemptCount() const { return attempt_count_; }

 private:
  GlobalLocalizerConfig cfg_;
  std::shared_ptr<GlobalRegistrationBackend> backend_;

  int confirmations_{0};
  Eigen::Isometry3d last_corroborated_{Eigen::Isometry3d::Identity()};
  bool has_corroborated_{false};
  std::size_t attempt_count_{0};
};

const char* toString(ProposalStatus s);
const char* toString(VerificationStatus s);

#ifdef SPARK_FAST_LIO_HAS_KISS_MATCHER
// Thin adapter over kiss_matcher::KISSMatcher. Kept in its own translation unit
// so that nothing else in the package needs KISS-Matcher's headers.
class KissMatcherBackend : public GlobalRegistrationBackend {
 public:
  explicit KissMatcherBackend(const GlobalLocalizerConfig& cfg);
  GlobalRegistrationResult align(const std::vector<Eigen::Vector3f>& source,
                                 const std::vector<Eigen::Vector3f>& target) override;
  std::string name() const override { return "kiss_matcher"; }

 private:
  GlobalLocalizerConfig cfg_;
};
#endif

}  // namespace reloc
}  // namespace spark_fast_lio
