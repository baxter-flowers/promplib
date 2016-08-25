from ..replayable import ReplayableInteractiveProMP as _ReplayableInteractiveProMP
from .bridge import ROSBridge
from numpy import mean

class ReplayableInteractiveProMP(_ReplayableInteractiveProMP):
    """
    Interactive ProMP that also stores the sequence of demos and goal requests for comparison
    ROS Overlay
    """
    def __init__(self, arm, epsilon_ok=0.03, with_orientation=True, min_num_demos=3, std_factor=2, dataset_id=-1):
        """
        :param arm: string ID of the FK/IK group (left, right, ...)
        :param epsilon_ok: maximum acceptable cartesian distance to the goal
        :param with_orientation: True for context = position + orientation, False for context = position only
        :param min_num_demos: Minimum number of demos per primitive
        :param std_factor: Factor applied to the cartesian standard deviation so within this range, the MP is valid
        :param dataset_id: ID of the dataset to work with, id < 0 will create a new one
        """
        super(ReplayableInteractiveProMP, self).__init__(arm, epsilon_ok, with_orientation, min_num_demos, std_factor, dataset_id)
        self._durations = []
        self.joint_names = []

    @property
    def mean_duration(self):
        return float(mean(self._durations))

    def add_demonstration(self, demonstration, eef_demonstration):
        """
        Add a new  demonstration for this skill and stores it into the current data set
        Automatically determine whether it is added to an existing a new ProMP
        :param demonstration: Joint-space demonstration demonstration[time][joint]
        :param eef_demonstration: Full end effector demo [[[x, y, z], [qx, qy, qz, qw]], [[x, y, z], [qx, qy, qz, qw]]...]
        :return: The ProMP id that received the demo
        """
        demonstration = ROSBridge.to_joint_trajectory(demonstration)
        if len(self.joint_names) > 0 and self.joint_names != demonstration.joint_names:
            raise ValueError("Joints must be the same and in same order for all demonstrations, this demonstration has joints {} while we had {}".format(demonstration.joint_names, self.joint_names))

        self._durations.append(demonstration.points[-1].time_from_start.to_sec() - demonstration.points[0].time_from_start.to_sec())
        self.joint_names = demonstration.joint_names
        demo_array = ROSBridge.trajectory_to_numpy(demonstration)
        eef_pose_array = ROSBridge.path_to_numpy(eef_demonstration)
        return super(ReplayableInteractiveProMP, self).add_demonstration(demo_array, eef_pose_array)

    def set_goal(self, x_des, joint_des=None):
        """
        Set a new task-space goal, and determine which primitive will be used
        :param x_des: desired task-space goal
        :param joint_des desired joint-space goal **ONLY used for plots**
        :return: True if the goal has been taken into account, False if a new demo is needed to reach it
        """
        joint_des = None if joint_des is None else ROSBridge.state_to_numpy(joint_des)
        return super(ReplayableInteractiveProMP, self).set_goal(x_des, joint_des)

    def generate_trajectory(self, force=False, duration=-1):
        trajectory_array = super(ReplayableInteractiveProMP, self).generate_trajectory(force)
        return ROSBridge.numpy_to_trajectory(trajectory_array, self.joint_names,
                                             duration if duration > 0 else self.mean_duration)

