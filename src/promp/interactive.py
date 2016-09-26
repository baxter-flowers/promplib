from numpy.linalg import norm
from .ik import FK, IK
from .qcartpromp import QCartProMP
from os.path import join

class InteractiveProMP(object):
    """
    Represents a single skill as a set of several multi-joint proMPs in joint space, each best suited for a specific area
    """
    def __init__(self, arm, epsilon_ok=0.03, with_orientation=True, min_num_demos=3, std_factor=2, path_plots=''):
        """
        :param arm: string ID of the FK/IK group (left, right, ...)
        :param epsilon_ok: maximum acceptable cartesian distance to the goal
        :param with_orientation: True for context = position + orientation, False for context = position only
        :param min_num_demos: Minimum number of demos per primitive
        :param std_factor: Factor applied to the cartesian standard deviation so within this range, the MP is valid
        :param path_plots: Path to output the plots, empty to disable
        """
        self.arm = arm
        self.promps = []
        self.fk = FK(arm)
        self.ik = IK(arm)
        self.epsilon_ok = epsilon_ok
        self.remaining_initial_demos = min_num_demos  # The first 3 demos will join the first primitive
        self.promp_write_index = -1  # Stores the index of the promp to be enriched, -1 in "new proMP" mode, -2 in "spontaneous demo" mode
        self.promp_read_index = -1  # Stores the index of the promp to be read for the previously set goal, -1 if no goal is known
        self.with_orientation = with_orientation
        self.min_num_demos = min_num_demos
        self.std_factor = std_factor
        self.goal_id = -1
        self.generated_trajectory = None
        self.path_plots = path_plots
        self.noise = {'position': 0.005, 'orientation': 0.05}


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

    def clear(self):
        self.promps = []
        self.remaining_initial_demos = self.min_num_demos
        self.promp_write_index = -1
        self.promp_read_index = -1
        self.goal_id = -1
        self.generated_trajectory = None

    def add_demonstration(self, demonstration, eef_demonstration, force_mp_target=-1):
        """
        Add a new  demonstration for this skill
        Automatically determine whether it is added to an existing a new ProMP
        :param demonstration: Joint-space demonstration demonstration[time][joint]
        :param eef_demonstration: Full end effector demo [[[x, y, z], [qx, qy, qz, qw]], [[x, y, z], [qx, qy, qz, qw]]...]
        :param force_mp_target: Force the target ProMP to the specified target ID
        :return: The ProMP id that received the demo (which is = to force_mp_target if force_mp_target >=0)
        """
        def select_best_promp(eef_demonstration):
            for promp_index, promp in enumerate(self.promps):
                if self._is_a_target(promp, eef_demonstration[-1]):
                    self.promp_write_index = promp_index
                    self.promp_read_index = -1  # This demo will enrich this promp
                    break  # Got one!
                else:
                    self.promp_write_index = -1
                    self.promp_read_index = -1  # This demo will end up to a new promp
                    # Don't break, search for a better one

        if force_mp_target > -1:
            self.promp_write_index = force_mp_target if self.num_primitives > force_mp_target else -1
        elif self.promp_write_index != -1 and self.promps[self.promp_write_index].num_demos >= self.min_num_demos:
            # This check ensures that after requesting a demo for this promp, we don't expect the user to provide such demo
            # So normal target search will occur, whatever a demo has been previously requested for a specific promp or not
            # Remove this check if we rely on the user to always provide a demo fot *that* promp if it has been requested
            self.promp_write_index = -1
            select_best_promp(eef_demonstration)
        # Search for a target MP to add this demo to
        elif self.promp_write_index == -1:
            select_best_promp(eef_demonstration)

        if self.promp_write_index == -1:
            # New ProMP requested
            self.promps.append(QCartProMP(self.arm, len(self.ik.joints), with_orientation=self.with_orientation, std_factor=self.std_factor,
                                          path_plots=join(self.path_plots, 'mp_' + str(len(self.promps)))))
            self.promp_write_index = self.num_primitives - 1

        # ProMP to be enriched identified, now add it the demonstration
        self.promps[self.promp_write_index].add_demonstration(demonstration, eef_demonstration[-1])
        return self.promp_write_index

    def need_demonstrations(self):
        """
        Get the needs in demonstrations to complete all the primitives with the minimum number of demonstrations
        :return: {promp_id: minimum_number_of_demos_remaining}
        """
        return {promp_id: self.min_num_demos - promp.num_demos for promp_id, promp in enumerate(self.promps) if promp.num_demos < self.min_num_demos}

    def _is_a_target(self, promp, goal):
        """
        Returns True whether the specified ProMP meets the requirements to be a possible target of the goal
        :param promp: Q-Cartesian pro MP object
        :param goal: [[x, y, z], [qx, qy, qz, qw]]
        :return: bool
        """
        std = promp.get_std_context()
        mean = promp.get_mean_context()
        for dimension in range(3):
            if not (mean[dimension] - self.std_factor*std[dimension] - self.noise['position']
                    < goal[0][dimension]
                    < mean[dimension] + self.std_factor*std[dimension] + self.noise['position']):
                return False

        if self.with_orientation:
            for dimension in range(4):
                if not (mean[dimension+3] - self.std_factor * std[dimension+3] - self.noise['orientation']
                        < goal[1][dimension]
                        < mean[dimension+3] + self.std_factor * std[dimension+3] + self.noise['orientation']):
                    return False
        return True

    def is_reached(self, trajectory, goal):
        """
        Returns True whether the specified trajectory actually reaches the goal with accepted precision
        :param trajectory: [[q1, q2, .. qn], [q1, q2, .. qn]]
        :param goal: [[x, y, z], [qx, qy, qz, qw]]
        :return: bool
        """
        reached_goal = self.fk.get(trajectory[-1])
        distance = norm(reached_goal[0] - goal[0])
        reached = distance < self.epsilon_ok
        print("Distance = {}m from goal".format(distance))
        return reached

    def set_goal(self, x_des, joint_des=None, refining=True):
        """
        Set a new task-space goal, and determine which primitive will be used
        :param x_des: desired task-space goal
        :param joint_des desired joint-space goal **ONLY used for plots**
        :param refining: True to refine the trajectory by optimization after conditioning
        :return: True if the goal has been taken into account, False if a new demo is needed to reach it
        """
        self.promp_write_index = -1
        self.goal_id += 1
        if self.num_primitives > 0:
            for promp_index, promp in enumerate(self.promps):
                if self._is_a_target(promp, x_des):
                    self.generated_trajectory = promp.generate_trajectory(x_des, refining, joint_des, 'set_goal_{}'.format(self.goal_id))
                    if self.is_reached(self.generated_trajectory, x_des):
                        print('MP {} goal {} is_reached=YES'.format(promp_index, self.goal_id))
                        self.goal = x_des
                        self.promp_read_index = promp_index
                        return True
                    else:
                        print('MP {} goal {} is_reached=NO'.format(promp_index, self.goal_id))
                        self.promp_write_index = promp_index
                        self.promp_read_index = -1  # A new demo is requested
                        return False
                else:
                    print('MP {} goal {} is_a_target=NO'.format(promp_index, self.goal_id))
                    _ = promp.generate_trajectory(x_des, refining, joint_des, 'set_goal_{}_not_a_target'.format(self.goal_id))  # Only for plotting
                    self.promp_read_index = -1  # A new promp is requested
            return False

    def generate_trajectory(self, force=False):
        if self.goal is None:
            raise RuntimeError("No goal set, use set_goal first")

        if not force and self.generated_trajectory is None:
            raise RuntimeError("No ProMP can reach this goal, use force=True to force")

        return self.generated_trajectory

    def plot_demos(self):
        for promp in self.promps:
            promp.plot_demos()