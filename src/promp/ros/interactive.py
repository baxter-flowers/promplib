from ..interactive import InteractiveProMP as _InteractiveProMP
from .bridge import ROSBridge
from numpy import mean


class InteractiveProMP(_InteractiveProMP):
    """
    Represents a single skill as a set of several multi-joint proMPs in joint space, each best suited for a specific area
    ROS Overlay
    """
    def __init__(self, arm, epsilon_ok=0.03, with_orientation=True, min_num_demos=3, std_factor=2):
        """
        :param arm: string ID of the FK/IK group (left, right, ...)
        :param epsilon_ok: maximum acceptable cartesian distance to the goal
        :param with_orientation: True for context = position + orientation, False for context = position only
        :param min_num_demos: Minimum number of demos per primitive
        """
        super(InteractiveProMP, self).__init__(arm, epsilon_ok, with_orientation, min_num_demos, std_factor)
        self._durations = []
        self.joint_names = []

    @property
    def mean_duration(self):
        return float(mean(self._durations))

    def add_demonstration(self, demonstration, eef_demonstration):
        """
        Add a new  demonstration for this skill
        Automatically determine whether it is added to an existing a new ProMP
        :param demonstration: Joint-space demonstration demonstration[time][joint]
        :param eef_demonstration: JointState or RobotState
        :return: The ProMP id that received the demo
        """
        demonstration = ROSBridge.to_joint_trajectory(demonstration)
        self._durations.append(demonstration.points[-1].time_from_start.to_sec() - demonstration.points[0].time_from_start.to_sec())
        self.joint_names = demonstration.joint_names
        return super(InteractiveProMP, self).add_demonstration(ROSBridge.trajectory_to_numpy(demonstration),
                                                               ROSBridge.path_to_numpy(eef_demonstration))

    def generate_trajectory(self, force=False, duration=-1):
        trajectory_array = super(InteractiveProMP, self).generate_trajectory(force)
        return ROSBridge.numpy_to_trajectory(trajectory_array, self.joint_names,
                                             duration if duration > 0 else self.mean_duration)

    def set_goal(self, x_des, joint_des=None):
        """
        Set a new task-space goal, and determine which primitive will be used
        :param x_des: desired task-space goal
        :param joint_des desired joint-space goal (RobotState) **ONLY used for plots**
        :return: True if the goal has been taken into account, False if a new demo is needed to reach it
        """
        np_joint_des = ROSBridge.state_to_numpy(joint_des) if joint_des is not None else None
        return super(InteractiveProMP, self).set_goal(x_des, np_joint_des)