import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    # Comma-separated robot namespaces, e.g. "husky,unitree2".
    robots = [r.strip() for r in
              LaunchConfiguration('robots').perform(context).split(',') if r.strip()]
    world = LaunchConfiguration('world_frame').perform(context)

    # world -> <ns>/map identity per robot. Ties each robot's map (KISS-Matcher for the
    # mapper, spark reloc for the relocalizer) into one shared world frame.
    nodes = []
    for ns in robots:
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'world_to_{ns}_map',
            output='screen',
            arguments=['--frame-id', world, '--child-frame-id', f'{ns}/map'],
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robots', default_value='husky,unitree2',
                              description='Comma-separated robot namespaces to link into world'),
        DeclareLaunchArgument('world_frame', default_value='world',
                              description='Shared root frame'),
        OpaqueFunction(function=launch_setup),
    ])
