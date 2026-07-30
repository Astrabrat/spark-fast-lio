import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    config_path = LaunchConfiguration('config_path').perform(context)
    rviz_path = LaunchConfiguration('rviz_path').perform(context)
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    # Non-empty status topic switches the mapper into relocalization mode (waits for the
    # reloc node, applies map<-odom correction). Kept out of the yaml so plain mapping
    # with the same config stays standalone.
    status_topic = LaunchConfiguration('status_topic').perform(context)

    lio_node = Node(
        package='spark_fast_lio',
        executable='spark_lio_mapping',
        name='lio_mapping',
        namespace=namespace,
        output='screen',
        on_exit=Shutdown(),
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        parameters=[config_path,
                    {'use_sim_time': use_sim_time,
                     'relocalization.status_topic': status_topic}],
    )

    reloc_node = Node(
        package='spark_fast_lio',
        executable='spark_lio_relocalization',
        name='lio_relocalization',
        namespace=namespace,
        output='screen',
        on_exit=Shutdown(),
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        parameters=[config_path,
                    {'use_sim_time': use_sim_time,
                     'relocalization.status_topic': status_topic}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        namespace=namespace,
        prefix='nice',
        output='screen',
        arguments=['-d', rviz_path],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('start_rviz')),
    )

    return [lio_node, reloc_node, rviz_node]


def generate_launch_description():
    pkg_share = get_package_share_directory('spark_fast_lio')
    default_config = os.path.join(pkg_share, 'config', 'unitree.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz', 'unitree_reloc.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='unitree',
                              description='Namespace for LIO topics (e.g. robot1)'),
        DeclareLaunchArgument('start_rviz', default_value='false',
                              description='automatically start rviz'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='use /clock from bag playback'),
        DeclareLaunchArgument('status_topic', default_value='fast_lio/relocalization_status',
                              description='reloc status topic; enables relocalization mode'),
        DeclareLaunchArgument('config_path', default_value=default_config,
                              description='Model-specific configuration'),
        DeclareLaunchArgument('rviz_path', default_value=default_rviz,
                              description='rviz file to load'),
        OpaqueFunction(function=launch_setup),
    ])
