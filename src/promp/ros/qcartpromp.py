from numpy import mean
from ..qcartpromp import QCartProMP as _QCartProMP
from .bridge import ROSBridge


class QCartProMP(_QCartProMP):
    def __init__(self, arm, num_joints=7, num_basis=20, sigma=0.05, noise=.0001, num_samples=100, with_orientation=True, std_factor=2):
        super(QCartProMP, self).__init__(arm, num_joints, num_basis, sigma, noise, num_samples, with_orientation, std_factor)
        self._durations = []
        self.joint_names = []

    @property
    def mean_duration(self):
        return float(mean(self._durations))

    def add_demonstration(self, demonstration, eef_pose):
        """
        Add a new  demonstration and update the model
        :param demonstration: RobotTrajectory or JointTrajectory object
        :param eef_pose: Path object of end effector
        :return:
        """
        demonstration = ROSBridge.to_joint_trajectory(demonstration)
        if len(self.joint_names) > 0 and self.joint_names != demonstration.joint_names:
            raise ValueError("Joints must be the same and in same order for all demonstrations, this demonstration has joints {} while we had {}".format(demonstration.joint_names, self.joint_names))

        self._durations.append(demonstration.points[-1].time_from_start.to_sec() - demonstration.points[0].time_from_start.to_sec())
        self.joint_names = demonstration.joint_names
        demo_array = ROSBridge.trajectory_to_numpy(demonstration)
        eef_pose_array = ROSBridge.path_to_numpy(eef_pose)
        super(QCartProMP, self).add_demonstration(demo_array, eef_pose_array[-1])

    def generate_trajectory(self, cartesian_goal, refine=True, goal_joint_state_plot=None, stamp='', duration=-1):
        """
        Generate a new trajectory from the given demonstrations and parameters
        :param cartesian_goal: [[x, y, z], [x, y, z, w]] Actual goal in cartesian space
        :param refine: True if the trajectory must be refined by optimization after generation
        :param stamp: string stamp for plot file names
        :param duration: Desired duration, auto if duration < 0
        :param goal_joint_state_plot: RobotState of the desired joint-space goal **for plotting/debugging only**
        :return: the generated RobotTrajectory message
        """
        joint_goal_plot = ROSBridge.state_to_numpy(goal_joint_state_plot) if goal_joint_state_plot is not None else None
        trajectory_array = super(QCartProMP, self).generate_trajectory(cartesian_goal, refine, joint_goal_plot, stamp)
        return ROSBridge.numpy_to_trajectory(trajectory_array, self.joint_names,
                                             float(self.mean_duration) if duration < 0 else duration)