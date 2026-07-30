#!/usr/bin/env python3
"""World relay (simple testing aggregator).

Each robot keeps its own namespaced tree on /<ns>/tf + /<ns>/tf_static. This node
forwards every robot's tree onto the GLOBAL /tf + /tf_static and adds the roots:

  world -> <ns>/map   static identity

so a single world-rooted RViz sees all robots. Safe because every frame is
uniquely namespaced (husky/odom != unitree2/odom) -> no collision on /tf.

NOTE: this republishes each robot's full high-rate tree onto one bus -- fine for
single-machine testing, not for a real multi-robot deployment.
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from tf2_msgs.msg import TFMessage

# tf2_ros uses these QoS for /tf and /tf_static; match them so we receive and
# re-serve latched statics correctly.
TF_QOS = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST)
TF_STATIC_QOS = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)


class WorldRelay(Node):
    def __init__(self):
        super().__init__('world_relay')
        self.world = self.declare_parameter('world_frame', 'world').value
        robots = self.declare_parameter('robots', ['husky', 'unitree2']).value

        self.pub_tf = self.create_publisher(TFMessage, '/tf', TF_QOS)
        self.pub_tf_static = self.create_publisher(TFMessage, '/tf_static', TF_STATIC_QOS)

        for ns in robots:
            self.create_subscription(
                TFMessage, f'/{ns}/tf',
                lambda m: self.pub_tf.publish(m), TF_QOS)
            self.create_subscription(
                TFMessage, f'/{ns}/tf_static',
                lambda m: self.pub_tf_static.publish(m), TF_STATIC_QOS)

        # world -> <ns>/map roots (latched, on global /tf_static).
        roots = [self._identity_tf(f'{ns}/map') for ns in robots]
        self.pub_tf_static.publish(TFMessage(transforms=roots))

        self.get_logger().info(
            f'world_relay up. world={self.world} robots={robots} '
            f'(forwarding /<ns>/tf[_static] -> global /tf[_static])')

    def _identity_tf(self, child):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world
        t.child_frame_id = child
        t.transform.rotation.w = 1.0
        return t


def main():
    rclpy.init()
    node = WorldRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
