// Offline harness for the global relocalization pipeline.
//
// Runs exactly the stages the node runs — KISS-Matcher global registration,
// GICP refinement against a cropped submap, inlier-ratio scoring — against a
// prior .pcd, with no ROS graph and no robot. Use it to pick voxel sizes and
// inlier thresholds for a given map before deploying them.
//
//   reloc_offline <prior.pcd> --query <scan.pcd>
//   reloc_offline <prior.pcd> --carve 12,-4,0,35
//
// With --carve, a query cloud is cut out of the prior map at a known pose and
// transformed by a known amount, so the recovered pose can be scored against
// ground truth. That is the mode to start with: if relocalization cannot
// recover a transform it applied to the map itself, the parameters are wrong
// and no amount of real data will help.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>

#include "global_localizer.h"
#include "reloc_core.h"

using namespace spark_fast_lio::reloc;

namespace {

constexpr double kDeg = M_PI / 180.0;

struct Args {
  std::string prior_path;
  std::string query_path;
  bool carve{false};
  double carve_x{0}, carve_y{0}, carve_z{0}, carve_yaw_deg{0};
  double carve_radius{30.0};
  double kiss_voxel{0.5};
  double map_voxel{0.5};
  double track_voxel{0.2};
  double query_voxel{0.0};  // 0 = as-is; the node applies scan_voxel_size here
  double submap_radius{60.0};
  double inlier_dist{0.5};
  std::size_t min_inliers{20};
  bool use_quatro{true};
  double gicp_corresp{3.0};
  bool have_initial{false};
  double init_x{0}, init_y{0}, init_z{0}, init_yaw_deg{0};
};

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " <prior.pcd> [options]\n"
      << "  --query <scan.pcd>       query cloud to localize (real data)\n"
      << "  --carve x,y,z,yaw_deg    synthesize a query from the map at a known pose\n"
      << "  --carve-radius <m>       radius of the carved query cloud (default 30)\n"
      << "  --kiss-voxel <m>         KISS-Matcher voxel size (default 0.5)\n"
      << "  --map-voxel <m>          prior-map leaf for global registration (default 0.5)\n"
      << "  --track-voxel <m>        prior-map leaf for GICP refinement (default 0.2)\n"
      << "  --query-voxel <m>        downsample the query (mirrors scan_voxel_size)\n"
      << "  --submap-radius <m>      GICP submap crop radius (default 60)\n"
      << "  --inlier-dist <m>        verification inlier distance (default 0.5)\n"
      << "  --min-inliers <n>        minimum KISS-Matcher final inliers (default 20)\n"
      << "  --gicp-corresp <m>       GICP max correspondence distance (default 3.0)\n"
      << "  --no-quatro              use full SO(3) GNC instead of yaw-decoupled Quatro\n"
      << "  --initial x,y,z,yaw_deg  skip global registration and refine from this pose\n"
      << "                           (mirrors the operator /initialpose path)\n";
}

bool parseArgs(int argc, char** argv, Args& a) {
  if (argc < 2) return false;
  a.prior_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];
    auto next           = [&](double& d) { d = std::atof(argv[++i]); };
    if (k == "--query" && i + 1 < argc) a.query_path = argv[++i];
    else if (k == "--carve" && i + 1 < argc) {
      a.carve = true;
      if (std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &a.carve_x, &a.carve_y, &a.carve_z,
                      &a.carve_yaw_deg) != 4) {
        std::cerr << "--carve expects x,y,z,yaw_deg\n";
        return false;
      }
    }
    else if (k == "--initial" && i + 1 < argc) {
      a.have_initial = true;
      if (std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &a.init_x, &a.init_y, &a.init_z,
                      &a.init_yaw_deg) != 4) {
        std::cerr << "--initial expects x,y,z,yaw_deg\n";
        return false;
      }
    }
    else if (k == "--carve-radius" && i + 1 < argc) next(a.carve_radius);
    else if (k == "--kiss-voxel" && i + 1 < argc) next(a.kiss_voxel);
    else if (k == "--map-voxel" && i + 1 < argc) next(a.map_voxel);
    else if (k == "--track-voxel" && i + 1 < argc) next(a.track_voxel);
    else if (k == "--query-voxel" && i + 1 < argc) next(a.query_voxel);
    else if (k == "--submap-radius" && i + 1 < argc) next(a.submap_radius);
    else if (k == "--inlier-dist" && i + 1 < argc) next(a.inlier_dist);
    else if (k == "--gicp-corresp" && i + 1 < argc) next(a.gicp_corresp);
    else if (k == "--min-inliers" && i + 1 < argc) a.min_inliers = std::atoi(argv[++i]);
    else if (k == "--no-quatro") a.use_quatro = false;
    else {
      std::cerr << "Unknown option: " << k << "\n";
      return false;
    }
  }
  return !a.prior_path.empty() && (a.carve || !a.query_path.empty());
}

Eigen::Isometry3d yawPose(double x, double y, double z, double yaw_deg) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(yaw_deg * kDeg, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T.translation() = Eigen::Vector3d(x, y, z);
  return T;
}

// Isolates the only KISS-Matcher-dependent step, so the rest of this tool
// (and its GICP/scoring path, which mirrors the node's) still compiles when
// the package is built without KISS-Matcher.
bool makeLocalizer(const GlobalLocalizerConfig& cfg, std::unique_ptr<GlobalLocalizer>& out) {
#ifdef SPARK_FAST_LIO_HAS_KISS_MATCHER
  out = std::make_unique<GlobalLocalizer>(cfg, std::make_shared<KissMatcherBackend>(cfg));
  return true;
#else
  (void)cfg;
  (void)out;
  return false;
#endif
}

void printPose(const char* label, const Eigen::Isometry3d& T) {
  const Eigen::Vector3d t  = T.translation();
  const Eigen::Matrix3d& R = T.linear();
  // Explicit ZYX decomposition. Eigen's eulerAngles() picks a canonical range
  // that can report an upright yaw of 0 as (180, 180, 180), which reads as a
  // catastrophic error when the pose is in fact correct.
  const double yaw   = std::atan2(R(1, 0), R(0, 0));
  const double pitch = std::atan2(-R(2, 0), std::hypot(R(2, 1), R(2, 2)));
  const double roll  = std::atan2(R(2, 1), R(2, 2));
  std::cout << std::fixed << std::setprecision(3) << label << " xyz=(" << t.x() << ", " << t.y()
            << ", " << t.z() << ")  ypr=(" << yaw / kDeg << ", " << pitch / kDeg << ", "
            << roll / kDeg << ") deg\n";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parseArgs(argc, argv, args)) {
    usage(argv[0]);
    return 2;
  }

  PointCloudT::Ptr raw(new PointCloudT);
  if (pcl::io::loadPCDFile<PointT>(args.prior_path, *raw) < 0) {
    std::cerr << "Failed to load prior map: " << args.prior_path << "\n";
    return 1;
  }
  std::cout << "Prior map: " << raw->size() << " points\n";

  PointCloudT::Ptr track_map = voxelDownsample(PointCloudT::ConstPtr(raw), args.track_voxel);
  PointCloudT::Ptr global_map = voxelDownsample(PointCloudT::ConstPtr(raw), args.map_voxel);
  const std::vector<Eigen::Vector3f> target_vec = toVector(*global_map);
  std::cout << "Global target: " << target_vec.size() << " pts (leaf " << args.map_voxel
            << ")   Tracking target: " << track_map->size() << " pts (leaf " << args.track_voxel
            << ")\n";

  // --- Build the query cloud ----------------------------------------------
  PointCloudT::Ptr query(new PointCloudT);
  bool have_truth = false;
  Eigen::Isometry3d truth_map_T_query = Eigen::Isometry3d::Identity();

  if (args.carve) {
    // Cut a sphere out of the map, then move it by a known transform. The
    // recovered map<-query should be the inverse of what we applied.
    const Eigen::Vector3d center(args.carve_x, args.carve_y, args.carve_z);
    PointCloudT::Ptr patch(new PointCloudT);
    for (const auto& p : raw->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      if ((Eigen::Vector3d(p.x, p.y, p.z) - center).norm() <= args.carve_radius) {
        patch->points.push_back(p);
      }
    }
    patch->width  = static_cast<uint32_t>(patch->points.size());
    patch->height = 1;
    if (patch->empty()) {
      std::cerr << "--carve produced an empty patch; is the centre inside the map?\n";
      return 1;
    }
    truth_map_T_query = yawPose(args.carve_x, args.carve_y, args.carve_z, args.carve_yaw_deg);
    have_truth        = true;
    // query = truth^-1 * patch, so that truth * query == patch (in the map).
    pcl::transformPointCloud(*patch, *query,
                             truth_map_T_query.inverse().matrix().cast<float>());
    std::cout << "Carved query: " << query->size() << " points around (" << args.carve_x << ", "
              << args.carve_y << ", " << args.carve_z << ") r=" << args.carve_radius << "\n";
    printPose("Ground truth map<-query:", truth_map_T_query);
  } else {
    if (pcl::io::loadPCDFile<PointT>(args.query_path, *query) < 0) {
      std::cerr << "Failed to load query cloud: " << args.query_path << "\n";
      return 1;
    }
    query = removeNonFinite(PointCloudT::ConstPtr(query));
    std::cout << "Query: " << query->size() << " points";
    if (args.query_voxel > 0.0) {
      query = voxelDownsample(PointCloudT::ConstPtr(query), args.query_voxel);
      std::cout << " -> " << query->size() << " after leaf " << args.query_voxel;
    }
    std::cout << "\n";
  }

  // --- Stage 1: global registration ---------------------------------------
  GlobalLocalizerConfig cfg;
  cfg.voxel_size              = static_cast<float>(args.kiss_voxel);
  cfg.use_quatro              = args.use_quatro;
  cfg.min_final_inliers       = args.min_inliers;
  cfg.min_source_points       = 100;
  cfg.min_verify_inlier_ratio = 0.0;  // report rather than gate, offline
  cfg.required_confirmations  = 1;

  Proposal p;
  if (args.have_initial) {
    std::cout << "\n--- Stage 1: SKIPPED (--initial given) ---\n";
    p.status     = ProposalStatus::kOk;
    p.map_T_odom = yawPose(args.init_x, args.init_y, args.init_z, args.init_yaw_deg);
  } else {
    std::unique_ptr<GlobalLocalizer> localizer;
    if (!makeLocalizer(cfg, localizer)) {
      std::cerr << "This binary was built without KISS-Matcher, so there is no global "
                   "registration backend.\nBuild KISS-Matcher's cpp/kiss_matcher and re-run "
                   "cmake on spark_fast_lio, or pass --initial to test stage 2 alone.\n";
      return 3;
    }
    std::cout << "\n--- Stage 1: KISS-Matcher global registration ---\n";
    p = localizer->propose(*query, target_vec);
    std::cout << "status=" << toString(p.status)
              << "  final_inliers=" << p.registration.num_final_inliers
              << "  rotation_inliers=" << p.registration.num_rotation_inliers << "  time="
              << std::setprecision(2) << p.registration.seconds << " s\n";
    if (!p.ok()) {
      std::cout << "\nRelocalization FAILED at stage 1.\n"
                << "Try a larger --kiss-voxel (map-scale registration usually wants 0.5-1.0 m), "
                << "a larger query extent, or --no-quatro if the sensor is not roughly level.\n";
      return 1;
    }
  }
  printPose("Proposed map<-query:", p.map_T_odom);

  // --- Stage 2: GICP refinement on a cropped submap ------------------------
  std::cout << "\n--- Stage 2: GICP refinement + verification ---\n";
  SubmapCropper cropper(track_map, args.submap_radius, 0.0);
  if (!cropper.update(p.map_T_odom.translation())) {
    std::cerr << "Submap crop failed.\n";
    return 1;
  }
  if (!cropper.valid()) {
    std::cout << "Proposal lands outside the prior map: no submap to refine against.\n";
    return 1;
  }
  std::cout << "Submap: " << cropper.submap()->size() << " points within " << args.submap_radius
            << " m\n";

  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
  gicp.setMaxCorrespondenceDistance(args.gicp_corresp);
  gicp.setMaximumIterations(50);
  gicp.setTransformationEpsilon(1e-4);
  gicp.setInputTarget(cropper.submap());
  gicp.setInputSource(PointCloudT::ConstPtr(query));

  PointCloudT aligned;
  gicp.align(aligned, p.map_T_odom.matrix().cast<float>());
  if (!gicp.hasConverged()) {
    std::cout << "GICP did not converge.\n";
    return 1;
  }

  Eigen::Isometry3d refined;
  refined.matrix()  = gicp.getFinalTransformation().cast<double>();
  refined           = orthonormalized(refined);
  const auto q      = evaluate(aligned, cropper.tree(), args.inlier_dist);

  printPose("Refined  map<-query:", refined);
  std::cout << "inlier_ratio=" << std::setprecision(3) << q.inlier_ratio << "  ("
            << q.num_inliers << "/" << q.num_evaluated << ")  mean_inlier_error="
            << q.mean_inlier_error << " m\n";
  std::cout << "pcl getFitnessScore()=" << gicp.getFitnessScore()
            << "  <- note how this moves with map coverage; inlier_ratio does not\n";

  if (have_truth) {
    const PoseDelta e_prop = poseDelta(p.map_T_odom, truth_map_T_query);
    const PoseDelta e_ref  = poseDelta(refined, truth_map_T_query);
    std::cout << "\n--- Error vs ground truth ---\n"
              << "after KISS-Matcher: " << e_prop.translation_m << " m, "
              << e_prop.rotation_rad / kDeg << " deg\n"
              << "after GICP        : " << e_ref.translation_m << " m, "
              << e_ref.rotation_rad / kDeg << " deg\n";
    const bool pass = e_ref.translation_m < 0.5 && e_ref.rotation_rad < 2.0 * kDeg;
    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
  }
  return 0;
}
