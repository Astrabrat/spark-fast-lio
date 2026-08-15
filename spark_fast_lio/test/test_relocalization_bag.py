"""Bag-driven integration test for prior-map relocalization.

Plays a recorded run against a prior .pcd and asserts that the node relocalizes
*without* any operator pose being published, then holds that lock. This is the
only test that exercises the real KISS-Matcher backend, the ROS wiring and the
TF output together; the gtest suites cover the decision logic in isolation.

The test needs real data, so it self-skips unless both are set:

    export RELOC_TEST_BAG=/path/to/rosbag2_dir
    export RELOC_TEST_MAP=/path/to/prior_map.pcd
    colcon test --packages-select spark_fast_lio

Optional:
    RELOC_TEST_CONFIG   config yaml (default: the installed husky.yaml)
    RELOC_TEST_NS       node namespace (default: husky)
    RELOC_TEST_TIMEOUT  seconds to wait for relocalization (default: 120)
    RELOC_EXPECT_XYZ    "x,y,z" the robot should end near, in the map frame
    RELOC_EXPECT_TOL    metres of tolerance for the above (default: 2.0)
    RELOC_BAG_RATE      ros2 bag play rate (default: 1.0)
"""

import os
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool

BAG = os.environ.get("RELOC_TEST_BAG", "")
MAP = os.environ.get("RELOC_TEST_MAP", "")
NS = os.environ.get("RELOC_TEST_NS", "husky")
TIMEOUT = float(os.environ.get("RELOC_TEST_TIMEOUT", "120"))
RATE = os.environ.get("RELOC_BAG_RATE", "1.0")

_SKIP_REASON = "set RELOC_TEST_BAG and RELOC_TEST_MAP to run the bag integration test"


def _config_path():
    override = os.environ.get("RELOC_TEST_CONFIG", "")
    if override:
        return override
    return os.path.join(get_package_share_directory("spark_fast_lio"), "config", "husky.yaml")


@pytest.mark.launch_test
def generate_test_description():
    if not BAG or not MAP:
        # launch_testing needs a valid description even when every test skips.
        return launch.LaunchDescription([launch_testing.actions.ReadyToTest()])

    common = [
        _config_path(),
        {
            "use_sim_time": True,
            "relocalization.map_file": MAP,
            "relocalization.status_topic": "fast_lio/relocalization_status",
            # The whole point of the test: no operator pose is ever sent.
            "relocalization.global.enabled": True,
            # Keep the test deterministic and reasonably quick.
            "relocalization.global.attempt_period": 2.0,
            "relocalization.global.required_confirmations": 2,
        },
    ]

    lio = launch_ros.actions.Node(
        package="spark_fast_lio",
        executable="spark_lio_mapping",
        name="lio_mapping",
        namespace=NS,
        output="screen",
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        parameters=common,
    )
    reloc = launch_ros.actions.Node(
        package="spark_fast_lio",
        executable="spark_lio_relocalization",
        name="lio_relocalization",
        namespace=NS,
        output="screen",
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        parameters=common,
    )
    bag = launch.actions.ExecuteProcess(
        cmd=["ros2", "bag", "play", BAG, "--clock", "--rate", RATE],
        output="screen",
    )

    return launch.LaunchDescription(
        [lio, reloc, bag, launch_testing.actions.ReadyToTest()]
    ), {"reloc_process": reloc, "bag_process": bag}


class TestRelocalizesWithoutOperatorPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("reloc_bag_test")

    def tearDown(self):
        self.node.destroy_node()

    @unittest.skipUnless(BAG and MAP, _SKIP_REASON)
    def test_status_latches_true_and_pose_is_published(self):
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        statuses = []
        odoms = []
        self.node.create_subscription(
            Bool, f"/{NS}/fast_lio/relocalization_status", statuses.append, latched
        )
        self.node.create_subscription(
            Odometry, f"/{NS}/fast_lio/relocalized_odometry", odoms.append, 20
        )

        # The node must publish status=False first, so that a stale latched
        # True from a previous run can never be mistaken for success here.
        deadline = self.node.get_clock().now().nanoseconds * 1e-9 + TIMEOUT
        relocalized = False
        while self.node.get_clock().now().nanoseconds * 1e-9 < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.5)
            if any(m.data for m in statuses):
                relocalized = True
                break

        self.assertTrue(
            relocalized,
            f"never relocalized within {TIMEOUT}s with no operator pose "
            f"(status messages seen: {[m.data for m in statuses]})",
        )
        self.assertTrue(statuses, "no status messages at all")
        self.assertFalse(
            statuses[0].data, "first status must be False, not a stale latched True"
        )

        # Let it run on a bit and confirm the lock holds and odometry flows.
        settle = self.node.get_clock().now().nanoseconds * 1e-9 + 10.0
        while self.node.get_clock().now().nanoseconds * 1e-9 < settle:
            rclpy.spin_once(self.node, timeout_sec=0.5)

        self.assertGreater(len(odoms), 0, "no localized odometry published after relocalizing")
        self.assertFalse(
            statuses[-1].data is False,
            "relocalization was lost again after latching",
        )

        expect = os.environ.get("RELOC_EXPECT_XYZ", "")
        if expect:
            tol = float(os.environ.get("RELOC_EXPECT_TOL", "2.0"))
            ex, ey, ez = (float(v) for v in expect.split(","))
            p = odoms[-1].pose.pose.position
            err = ((p.x - ex) ** 2 + (p.y - ey) ** 2 + (p.z - ez) ** 2) ** 0.5
            self.assertLess(
                err,
                tol,
                f"relocalized to ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}), "
                f"expected ({ex}, {ey}, {ez}) within {tol} m — "
                "this is the aliasing failure mode, not a tuning issue",
            )


@launch_testing.post_shutdown_test()
class TestNodeShutdownCleanly(unittest.TestCase):
    @unittest.skipUnless(BAG and MAP, _SKIP_REASON)
    def test_reloc_exits_without_error(self, proc_info, reloc_process):
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -2, -15], process=reloc_process
        )
