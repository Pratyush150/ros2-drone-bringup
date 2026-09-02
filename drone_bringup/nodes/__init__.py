"""rclpy nodes.

Every module in here imports ``rclpy`` at module scope on purpose -- these are
executables launched by ROS 2, and a silent fallback would hide a broken
install. The logic they drive lives in :mod:`drone_bringup.core`, which has no
ROS dependency at all.
"""
