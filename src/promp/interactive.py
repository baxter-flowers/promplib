from numpy.linalg import norm
from .ros import FK, IK
from .ros import QCartProMP
from moveit_msgs.msg import RobotState

class InteractiveProMP(object):
    """
    Represents a single skill as a set of several multi-joint proMPs in joint space, each best suited for a specific area
    """
    def __init__(self, arm, epsilon_ok=0.1, with_orientation=True, min_num_demos=3, std_factor=2):
        """
        :param arm: string ID of the FK/IK group (left, right, ...)
        :param epsilon_ok: maximum acceptable cartesian distance to the goal
        :param epsilon_new_demo: maximum cartesian distance to enrich an existing promp instead of creating a new
        :param with_orientation: True for context = position + orientation, False for context = position only
        :param min_num_demos: Minimum number of demos per primitive
        """
        self.promps = []
        self.fk = FK(arm)
        self.ik = IK(arm)
        self.epsilon_ok = epsilon_ok
        self.remaining_initial_demos = 3  # The first 3 demos will join the first primitive
        self.promp_write_index = -1  # Stores the index of the promp to be enriched, -1 in "new proMP" mode, -2 in "spontaneous demo" mode
        self.promp_read_index = -1  # Stores the index of the promp to be read for the previously set goal, -1 if no goal is known
        self.with_orientation = with_orientation
        self.min_num_demos = min_num_demos
        self.std_factor = std_factor

    @property
    def num_joints(self):
        return self.promps[0].num_joints if self.num_primitives > 0 else 0

    @property
    def num_demos(self):
        return sum([promp.num_demos for promp in self.promps])

    @property
    def num_points(self):
        return self.promps[0].num_points if self.num_primitives > 0 else 0

    @property
    def num_primitives(self):
        return len(self.promps)

    @property
    def status_writing(self):
        if self.promp_write_index == -1:
            return "new demonstration requested for a new Pro MP"
        elif self.promp_write_index == -2:
            return "spontaneous demo possible but not required"
        elif 0 <= self.promp_write_index < self.num_primitives:
            return "new demonstration requested for Pro MP {}".format(self.promp_write_index)
        return "unknown"

    @property
    def status_reading(self):
        if self.promp_read_index == -1:
            return "no goal set (or invalid)"
        elif 0 <= self.promp_read_index < self.num_primitives:
            return "goal is ready to be generated with Pro MP {}".format(self.promp_read_index)
        return "unknown"

    @staticmethod
    def _last_point_of_path(path):
        return [[path.poses[-1].pose.position.x,
                 path.poses[-1].pose.position.y,
                 path.poses[-1].pose.position.z],
                [path.poses[-1].pose.orientation.x,
                 path.poses[-1].pose.orientation.y,
                 path.poses[-1].pose.orientation.z,
                 path.poses[-1].pose.orientation.w]]

    def add_demonstration(self, demonstration, eef_demonstration):
        """
        Add a new  demonstration for this skill
        Automatically determine whether it is added to an existing a new ProMP
        :param demonstration: Joint-space demonstration RobotTrajectory or JointTrajectory object
        :param eef_demonstration: Path object of the end effector trajectory corresponding to the joint demo
        :return:
        """
        if self.promp_write_index == -1:   # Do not override this setting, the mp might have been forced to reach the minimum number of demos
            for promp_index, promp in enumerate(self.promps):
                last_point = self._last_point_of_path(eef_demonstration)
                if self._is_a_target(promp, last_point):
                    self.promp_write_index = promp_index
                    self.promp_read_index = -1  # This demo will enrich this promp
                    break  # Got one!
                else:
                    self.promp_write_index = -1
                    self.promp_read_index = -1  # This demo will end up to a new promp
                    # Don't break, search for a better one

        if self.promp_write_index == -1:
            # New ProMP requested
            self.promps.append(QCartProMP(len(self.ik.joints), with_orientation=self.with_orientation, std_factor=self.std_factor))
            self.promp_write_index = self.num_primitives - 1

        # ProMP to be enriched identified, now add it the demonstration
        self.promps[self.promp_write_index].add_demonstration(demonstration, eef_demonstration)

    def need_demonstrations(self):
        """
        Get the needs in demonstrations to complete all the primitives with the minimum number of demonstrations
        :return: {promp_id: minimum_number_of_demos_remaining}
        """
        return {promp_id: self.min_num_demos - promp.num_demos for promp_id, promp in enumerate(self.promps) if promp.num_demos < self.min_num_demos}

    @staticmethod
    def _get_mean_std_position(promp):
        std = promp.promp.get_std_context()
        mean = promp.promp.get_mean_context()
        position_std = std[:3]
        position_mean = mean[:3]
        return position_mean, position_std

    def _is_a_target(self, promp, goal):
        """
        Returns True whether the specified ProMP meets the requirements to be a possible target of the goal
        :param promp:
        :param goal:
        :return: bool
        """
        position_mean, position_std = self._get_mean_std_position(promp)
        for dimension in range(3):
            if not position_mean[dimension] - self.std_factor*position_std[dimension]\
                    < goal[0][dimension]\
                    < position_mean[dimension] + self.std_factor*position_std[dimension]:
                return False
        return True

    @staticmethod
    def _last_point_to_state(trajectory):
        rs = RobotState()
        rs.joint_state.name = trajectory.joint_trajectory.joint_names
        rs.joint_state.position = trajectory.joint_trajectory.points[-1].positions
        return rs

    def is_reached(self, trajectory, goal):
        reached_goal = self.fk.get(self._last_point_to_state(trajectory))
        distance = norm(reached_goal[0] - goal[0])
        reached = distance < self.epsilon_ok
        print("Distance = {}m from goal".format(distance))
        return reached

    def set_goal(self, x_des):
        """
        Set a new task-space goal, and determine which primitive will be used
        :param x_des: desired task-space goal
        :return: True if the goal has been taken into account, False if a new demo is needed to reach it
        """
        self.promp_write_index = -1
        if self.num_primitives > 0:
            for promp_index, promp in enumerate(self.promps):
                if self._is_a_target(promp, x_des):
                    trajectory = promp.generate_trajectory(x_des)
                    if self.is_reached(trajectory, x_des):
                        self.goal = x_des
                        self.promp_read_index = promp_index
                        return True
                    else:
                        self.promp_write_index = promp_index
                        self.promp_read_index = -1  # A new demo is requested
                        return False
                else:
                    self.promp_read_index = -1  # A new promp is requested
            return False

    def generate_trajectory(self, force=False, duration=-1):
        if self.goal is None:
            raise RuntimeError("No goal set, use set_goal first")

        if not force and self.promp_read_index < 0:
            raise RuntimeError("No ProMP can reach this goal, use force=True to force")

        return self.promps[self.promp_read_index].generate_trajectory(self.goal, duration)

    def plot(self, eef, is_goal=False):
        for promp_id, promp in enumerate(self.promps):
            promp.plot(eef, str(promp_id), is_goal)