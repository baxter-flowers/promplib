from ..replayable import ReplayableInteractiveProMP as _ReplayableInteractiveProMP
from .bridge import ROSBridge
from numpy import mean
from os.path import join
import rospkg
import json


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
        rospack = rospkg.RosPack()
        self.path_ds = join(rospack.get_path('prompros'), 'datasets')
        self.path_plots = join(rospack.get_path('prompros'), 'plots')
        super(ReplayableInteractiveProMP, self).__init__(arm, epsilon_ok, with_orientation, min_num_demos, std_factor, self.path_ds, dataset_id, self.path_plots)
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

        with open(join(self.dataset_path, 'durations.json'), 'w') as f:
            json.dump(self._durations, f)

        with open(join(self.dataset_path, 'joint_names.json'), 'w') as f:
            json.dump(self.joint_names, f)

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

    def play(self, keep_targets=True, refining=True):
        """
        Play all the events of this trajectory from start (add demo, set goals, ...)
        :param keep_targets: True to replay the sequence by respecting the same MP targets, False to recompute the best target
        :param refining: True to enable post-process refining, False to disable
        :return: the timeline of results of these goal or demo events
        """
        timeline = super(ReplayableInteractiveProMP, self).play(keep_targets, refining)

        with open(join(self.dataset_path, 'durations.json')) as f:
            durations = json.load(f)
        with open(join(self.dataset_path, 'joint_names.json')) as f:
            joint_names = json.load(f)

        for event in timeline:
            if event['type'] == 'goal' and event['is_reached']:
                event['trajectory'] = ROSBridge.numpy_to_trajectory(event['trajectory'], joint_names, float(mean(durations)))
        return timeline
