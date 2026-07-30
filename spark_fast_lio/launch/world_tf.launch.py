import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    robots = [r.strip() for r in
              LaunchConfiguration('robots').perform(context).split(',') if r.strip()]
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'

    # Base-station world relay: world->map statics + world->base from each robot's
    # relocalized pose.
    relay = Node(
        package='spark_fast_lio',
        executable='world_relay.py',
        name='world_relay',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'world_frame': LaunchConfiguration('world_frame').perform(context),
            'robots': robots,
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='world_rviz',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_path').perform(context)],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('start_rviz')),
    )
    return [relay, rviz]


def generate_launch_description():
    pkg_share = get_package_share_directory('spark_fast_lio')
    default_rviz = os.path.join(pkg_share, 'rviz', 'world.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('robots', default_value='husky,unitree',
                              description='Comma-separated robot namespaces to link into world'),
        DeclareLaunchArgument('world_frame', default_value='world',
                              description='Shared root frame'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='use /clock from bag playback'),
        DeclareLaunchArgument('start_rviz', default_value='true',
                              description='open the world-rooted mission RViz'),
        DeclareLaunchArgument('rviz_path', default_value=default_rviz,
                              description='rviz file to load'),
        OpaqueFunction(function=launch_setup),
    ])
