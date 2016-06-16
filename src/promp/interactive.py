from numpy import mean
from numpy.linalg import norm
from .ros import FK, IK
from .ros import ProMP


class InteractiveProMP(object):
    """
    Represents a single skill as a set of several multi-joint proMPs in joint space, each best suited for a specific area
    """
    def __init__(self, arm, epsilon_ok=0.01, epsilon_new_demo=0.3):
        """
        :param arm: string ID of the FK/IK group (left, right, ...)
        :param epsilon_ok: maximum acceptable cartesian distance to the goal
        :param epsilon_new_demo: maximum cartesian distance to enrich an existing promp instead of creating a new
        """
        self.promps = []
        self.fk = FK(arm)
        self.ik = IK(arm)
        self.epsilon_ok = epsilon_ok
        self.epsilon_new_demo = epsilon_new_demo
        self.promp_write_index = -1  # Stores the index of the promp to be enriched, -1 in "new proMP" mode, -2 in "spontaneous demo" mode
        self.promp_read_index = -1  # Stores the index of the promp to be read for th goal previously set, -1 if no goal is known

    @property
    def num_joints(self):
        return self.promps[0].num_joints if len(self.promps) > 0 else 0

    @property
    def num_demos(self):
        return sum([promp.num_demos for promp in self.promps])

    @property
    def num_points(self):
        return self.promps[0].num_points if len(self.promps) > 0 else 0

    @property
    def num_primitives(self):
        return len(self.promps)

    @property
    def status_writing(self):
        if self.promp_write_index == -1:
            return "new demonstration requested for a new promp"
        elif self.promp_write_index == -2:
            return "ready to set goal or add spontaneous demo"
        elif 0 <= self.promp_write_index < self.num_primitives:
            return "new demonstration requested for promp {}".format(self.promp_write_index)
        return "unknown"

    @property
    def status_reading(self):
        if self.promp_read_index == -1:
            return "goal is invalid"
        elif 0 <= self.promp_read_index < self.num_primitives:
            return "goal is ready to be generated with promp {}".format(self.promp_read_index)
        return "unknown"

    def clear_goal(self):
        for promp in self.promps:
            promp.clear_viapoints() # TODO this erases also non-goal viapoints, replace only the goal?
        self.promp_read_index = -1

    def add_demonstration(self, demonstration):
        """
        Add a new  demonstration for this skill
        Automatically determine whether it is added to an existing a new ProMP
        :param demonstration: RobotTrajectory or JointTrajectory object
        :return:
        """
        if self.promp_write_index == -2:
            # Spontaneous demonstration
            # If any promp has a distance < epsilon_new_demo, take the closest and enrich it
            min_distance = self.epsilon_new_demo
            for promp_index, promp in enumerate(self.promps):
                distance, goal_js = self._get_distance_and_goal(promp, self.fk.get(demonstration))
                if distance < min_distance:
                    min_distance = distance
                    self.promp_write_index = promp_index

        if self.promp_write_index == -1:
            # New ProMP requested
            self.promps.append(ProMP(len(self.ik.joints)))
            self.promp_write_index = len(self.promps) - 1

        # ProMP to be enriched identified, now add it the demonstration
        self.promps[self.promp_write_index].add_demonstration(demonstration)
        self.clear_goal()  # Invalidate the goal if any, the necessary ProMP might have changed
        self.promp_write_index = -2  # Go back in "spontaneous demo" mode

    @staticmethod
    def _distance_position(pose1, pose2):
        return norm(pose1[0] - pose2[0])

    def set_goal(self, x_des):
        """
        :param x_des:
        :return: (status, index) where the promp_index to add a new trajectory to if >=0, or -1 if a new
        """
        for promp_index, promp in enumerate(self.promps):
            distance, goal_js = self._get_distance_and_goal(promp, x_des)
            if distance < self.epsilon_ok:
                promp.clear_viapoints()  # TODO this erases also non-goal viapoints, replace only the goal?
                promp.set_goal(goal_js)
                self.promp_read_index = promp_index
                return True
            elif distance < self.epsilon_new_demo:
                self.promp_write_index = promp_index
                self.promp_read_index = -1  # A new demo has been requested
                return False
            else:
                self.promp_write_index = -1
                self.promp_read_index = -1  # A new promp has been requested
                return False

    def _get_distance_and_goal(self, promp, x_des):
        """
        Return the distance and joint space goal of the given promp from x_des
        """
        bounds = [promp.goal_bounds[joint] for joint in self.ik.joints]  # Reordering in case order in IK doesn't match promp order
        goal_js = self.ik.get(x_des, bounds=bounds)
        x_tilde = self.fk.get(goal_js[1])
        distance = self._distance_position(x_tilde, x_des)
        return distance if goal_js[0] else float('inf'), goal_js[1]

    def generate_trajectory(self, randomness=1e-10, duration=-1):
        """
        Generate a new trajectory from the given demonstrations and parameters
        :param x_des: Desired goal [[x, y, z], [x, y, z, w]]
        :param duration: Desired duration, auto if duration < 0
        :return: the generated RobotTrajectory message OR
        """
        if self.promp_read_index < 0:
            raise RuntimeError("Attempted to generated a trajectory while a new demonstration has been requested")
        elif self.promp_read_index < len(self.promps):
            return self.promps[self.promp_read_index].generate_trajectory(randomness, duration)
        else:
            raise RuntimeError("InteractiveProMP.promp_read_index has an unexpected value out of bounds")

