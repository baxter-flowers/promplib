from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory, RobotState
from nav_msgs.msg import Path
from rospy import Duration


class ROSBridge(object):
    @staticmethod
    def to_joint_trajectory(trajectory):
        if isinstance(trajectory, RobotTrajectory):
            trajectory = trajectory.joint_trajectory
        elif not isinstance(trajectory, JointTrajectory):
            raise TypeError("ROSBridge.to_joint_trajectory only accepts RT or JT, got {}".format(type(trajectory)))
        return trajectory

    @staticmethod
    def to_joint_state(state):
        if isinstance(state, RobotState):
            state = state.joint_state
        elif not isinstance(state, JointState):
            raise TypeError("ROSBridge.to_joint_trajectory only accepts RT or JT, got {}".format(type(trajectory)))
        return state

    @staticmethod
    def numpy_to_trajectory(trajectory, joint_names, duration):
        rt = RobotTrajectory()
        rt.joint_trajectory.joint_names = joint_names
        for point_idx, point in enumerate(trajectory):
            time = point_idx*duration/float(len(trajectory))
            jtp = JointTrajectoryPoint(positions=map(float, point), time_from_start=Duration(time))
            rt.joint_trajectory.points.append(jtp)
        return rt

    @staticmethod
    def pose_to_list(pose):
        if isinstance(pose, PoseStamped):
            plist = [[pose.pose.position.x,
                      pose.pose.position.y,
                      pose.pose.position.z],
                     [pose.pose.orientation.x,
                      pose.pose.orientation.y,
                      pose.pose.orientation.z,
                      pose.pose.orientation.w]]
            return plist
        raise TypeError("ROSBridge.pose_to_list only accepts Path, got {}".format(type(pose)))

    @staticmethod
    def path_last_point_to_numpy(path):
        if isinstance(path, Path):
            path = path.poses[-1]
        if isinstance(path, PoseStamped):
            return ROSBridge.pose_to_list(path)
        raise TypeError("ROSBridge.path_last_point_to_numpy only accepts Path or PoseStamped, got {}".format(type(path)))

    @staticmethod
    def trajectory_last_point_to_state(trajectory):
        trajectory = ROSBridge.to_joint_trajectory(trajectory)
        rs = RobotState()
        rs.joint_state.name = trajectory.joint_names
        rs.joint_state.position = trajectory.points[-1].positions
        return rs

    @staticmethod
    def path_to_numpy(path):
        path_list = []
        if isinstance(path, Path):
            for pose in path.poses:
                path_list.append(ROSBridge.pose_to_list(pose))
            return path_list
        raise TypeError("ROSBridge.path_to_numpy only accepts Path, got {}".format(type(path)))

    @staticmethod
    def trajectory_to_numpy(trajectory):
        trajectory = ROSBridge.to_joint_trajectory(trajectory)
        return [jtp.positions for jtp in trajectory.points]

    @staticmethod
    def state_to_numpy(state):
        state = ROSBridge.to_joint_state(state)
        return state.position
