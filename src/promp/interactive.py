from numpy.linalg import norm
from .ros import FK, IK
from .ros import QCartProMP

class InteractiveProMP(object):
    """
    Represents a single skill as a set of several multi-joint proMPs in joint space, each best suited for a specific area
    """
    def __init__(self, arm, epsilon_ok=0.01, epsilon_new_demo=0.3, with_orientation=True):
        """
        :param arm: string ID of the FK/IK group (left, right, ...)
        :param epsilon_ok: maximum acceptable cartesian distance to the goal
        :param epsilon_new_demo: maximum cartesian distance to enrich an existing promp instead of creating a new
        :param with_orientation: True for context = position + orientation, False for context = position only
        """
        self.promps = []
        self.fk = FK(arm)
        self.ik = IK(arm)
        self.epsilon_ok = epsilon_ok
        self.epsilon_new_demo = epsilon_new_demo
        self.remaining_initial_demos = 3  # The first 3 demos will join the first primitive
        self.promp_write_index = -1  # Stores the index of the promp to be enriched, -1 in "new proMP" mode, -2 in "spontaneous demo" mode
        self.promp_read_index = -1  # Stores the index of the promp to be read for the previously set goal, -1 if no goal is known
        self.with_orientation = with_orientation

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

    def clear_goal(self):
        for promp in self.promps:
            promp.clear_viapoints()  # TODO this erases also non-goal viapoints, replace only the goal?
        self.promp_read_index = -1
        self.promp_write_index = -1 if self.num_primitives == 0 else -2

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
        if self.remaining_initial_demos > 0 and self.num_primitives == 1:
            # Initial behaviour: The first demos are forced to join the same primitive
            self.promp_write_index = 0
            self.remaining_initial_demos -= 1
        else:
            # Normal behaviour will seek for a target proMP approaching the end eef point
            self.promp_write_index = -1
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
            self.promps.append(QCartProMP(len(self.ik.joints), with_orientation=self.with_orientation))
            self.promp_write_index = self.num_primitives - 1

        # ProMP to be enriched identified, now add it the demonstration
        self.promps[self.promp_write_index].add_demonstration(demonstration, eef_demonstration)
        self.clear_goal()  # Invalidate the goal if any, the necessary ProMP might have changed

    @staticmethod
    def _get_mean_cov_position(promp):
        cov = promp.promp.get_cov_context()
        mean = promp.promp.get_mean_context()
        position_cov = cov[:3,:3]
        position_mean = mean[:3]
        return position_mean, position_cov

    def _is_a_target(self, promp, goal):
        """
        Returns True whether the specified ProMP meets the requirements to be a possible target of the goal
        :param promp:
        :param goal:
        :return: bool
        """
        position_mean, position_cov = self._get_mean_cov_position(promp)
        for dimension in range(3):
            if not position_mean[dimension] - 2*position_cov[dimension][dimension]\
                    < goal[0][dimension]\
                    < position_mean[dimension] + 2*position_cov[dimension][dimension]:
                return False
        return True

    def _is_reachable(self, promp, goal):
        """
        Returns True whether the specified ProMP is able to reach the goal with enough precision
        :param promp:
        :param goal:
        :return: bool
        """
        mean, cov = promp.gaussian_conditioning_context(goal)
        position_mean = mean[:3]
        return norm(position_mean - goal[0]) < self.epsilon_ok

    def set_goal(self, x_des):
        """
        Set a new task-space goal, and determine which primitive will be used
        :param x_des: desired task-space goal
        :return: True if the goal has been taken into account, False if a new demo is needed to reach it
        """
        self.clear_goal()

        if self.num_primitives > 0:
            for promp_index, promp in enumerate(self.promps):
                if self._is_a_target(promp, x_des):
                    if self._is_reachable(promp, x_des):
                        self.promps[promp_index].set_goal(x_des)
                        self.promp_read_index = promp_index
                        return True
                    else:
                        self.promp_write_index = promp_index
                        self.promp_read_index = -1  # A new demo is requested
                        return False
                else:
                    self.promp_write_index = -1
                    self.promp_read_index = -1  # A new promp is requested
                    return False

    def generate_trajectory(self, force=False, randomness=1e-10, duration=-1):
        """
        Generate a new trajectory from the given demonstrations and parameters
        :param force: Force generation even if a demo has been requested to reach it
        :param duration: Desired duration, auto if duration < 0
        :return: the generated RobotTrajectory message
        """
        if self.promp_read_index < 0:
            if force:
                # if we have to provide a demo for a known primitive, generation can be forced using the writing index
                if self.promp_write_index < 0:
                    raise RuntimeError("Attempted to force trajectory generation but no suitable primitive has been found")
                else:
                    return self.promps[self.promp_write_index].generate_trajectory(randomness, duration)
            else:
                raise RuntimeError( "Attempted to generate a trajectory while a new demonstration has been requested (use force=True to go to the closest point)")
        elif self.promp_read_index < self.num_primitives:
            return self.promps[self.promp_read_index].generate_trajectory(randomness, duration)
        else:
            raise RuntimeError("InteractiveProMP.promp_read_index has an unexpected value out of bounds")

