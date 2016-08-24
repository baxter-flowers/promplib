import numpy as np
from scipy.interpolate import interp1d
from os.path import join, exists
from os import makedirs
import matplotlib.pyplot as plt



class QCartProMP(object):
    """
    n-dimensional probabilistic MP storing joints (Q) and end effector (Cart)
    """
    def __init__(self, num_joints=7, num_basis=20, sigma=0.05, noise=.0001, num_samples=100, with_orientation=True, std_factor=2, path_plots='/tmp/plots', mp_id=-1):
        self.num_basis = num_basis
        self.nrTraj = num_samples
        self.with_orientation = with_orientation
        self.context_length = 7 if with_orientation else 3
        self.std_factor = std_factor

        self.mu = np.linspace(0, 1, self.num_basis)
        self.z = np.linspace(0, 1, self.nrTraj).reshape((self.nrTraj, 1))
        self.noise = noise
        self.sigma = sigma * np.ones(self.num_basis)
        self.Gn = self._generate_basis_fx(self.z, self.mu, self.sigma)

        self.my_linRegRidgeFactor = 1e-8 * np.ones((self.num_basis, self.num_basis))
        self.MPPI = np.dot(np.linalg.inv(np.dot(self.Gn.T, self.Gn) + self.my_linRegRidgeFactor), self.Gn.T)
        self.Y = np.empty((0, num_samples, num_joints), float)  # all demonstrations
        self.x = np.linspace(0, 1, num_samples)

        self.W_full = np.zeros((0, self.num_joints * self.num_basis + self.context_length))
        self.contexts = []
        self.goal = None

        # number of variables to infer are the weights of the trajectories minus the number of context variables
        self.nQ = self.num_basis * self.num_joints

        self.mean_W_full = np.zeros(self.W_full.shape[1])
        self.cov_W_full = self.noise * np.ones((self.W_full.shape[1], self.W_full.shape[1]))

        self.plots = join(path_plots, str(mp_id)) if mp_id > -1 else path_plots
        self.plotted_points = []
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange']
        self.goal_id = 0

    def _generate_basis_fx(self, z, mu, sigma):
        z_minus_center = z - mu
        at = z_minus_center * (1. / sigma)  # pair-wise *
        # computing and normalizing basis (order 0)
        basis = np.exp(-0.5 * at ** 2) * (1. / (sigma * np.sqrt(2 * np.pi)))
        basis_sum = np.sum(basis, 1).reshape(self.nrTraj, 1)
        basis_n = basis * (1. / basis_sum)
        return basis_n

    def get_mean_context(self):
        return self.mean_W_full[-self.context_length:]

    def get_cov_context(self):
        return self.cov_W_full[-self.context_length:, -self.context_length:]

    def get_std_context(self):
        return np.sqrt(np.diagonal(self.get_cov_context()))

    def get_mean_w(self):
        return self.mean_W_full[:-self.context_length]

    def get_cov_w(self):
        return self.cov_W_full[:-self.context_length, :-self.context_length]

    def get_std_w(self):
        return np.sqrt(np.diagonal(self.get_cov_w()))

    def get_mean_joints(self):
        output = []
        for joint in range(self.num_joints):
            output.append(np.dot(self.Gn, self.get_mean_w()[joint*self.num_basis:(joint + 1) * self.num_basis]))
        return np.array(output)

    def get_std_joints(self):
        output = []
        for joint in range(self.num_joints):
            output.append(np.dot(self.Gn, self.get_std_w()[joint*self.num_basis:(joint + 1) * self.num_basis]))
        return np.array(output)

    def gaussian_conditioning_joints(self, goal):
        """
        Condition a temporary goal (not set) and return the mean and covariance of the joints
        :param goal: [[x, y, z], [x, y, z, w]]
        :return: (mean, cov)
        """

        # This is an alternative way to condition the ProMP.
        # It is simple compared to the original ProMP because it does not introduce
        # the features. This is possible because we are conditioning the ProMP on
        # the context, which in this case does not have a feature (feature is
        # identity). This function should not be used if you want to implement the
        # "real" ProMP.

        model = {}
        model["Cov11"] = self.cov_W_full[:self.nQ, :self.nQ]
        model["Cov12"] = self.cov_W_full[:self.nQ, self.nQ:]
        model["Cov21"] = self.cov_W_full[self.nQ:, :self.nQ]
        model["Cov22"] = self.cov_W_full[self.nQ:, self.nQ:]

        inv_context = np.linalg.inv(model["Cov22"] + self.noise * np.eye(model["Cov22"].shape[0]))
        obs = np.hstack(goal) if self.with_orientation else goal[0]
        mean_wNew = self.mean_W_full[:self.nQ] + np.dot(model["Cov12"], np.dot(inv_context, obs - self.mean_W_full[self.nQ:]))
        Cov_wNew = model["Cov11"] - np.dot(model["Cov12"], np.dot(inv_context, model["Cov21"]))

        return mean_wNew, Cov_wNew

    def dist_to_mean(self, q):
        return sum(abs(self.get_mean_context()[-4:] - q))

    def add_demonstration(self, demonstration, eef_pose):
        """
        Add a joint space demonstration and a task-space final end effector constraint
        :param demonstration: demonstration[point][joint]
        :param eef_pose: final end effector pose [[x, y, z], [x, y, z, w]]
        :return:
        """
        demonstration = np.array(demonstration)
        if demonstration.shape[1] != self.num_joints:
            raise ValueError("The given demonstration has {} joints while num_joints={}".format(demonstration.shape[1],
                                                                                                self.num_joints))

        # Strech the demo so that the new ones have the same number of points than previous demos and apply MPPI
        joint_demos = demonstration.T
        stretched_demos = []
        for joint_demo in joint_demos:
            interpolate = interp1d(np.linspace(0, 1, len(joint_demo)), joint_demo, kind='cubic')
            stretched_demo = interpolate(self.x)
            stretched_demos.append(stretched_demo)
        stretched_demos = np.array(stretched_demos)
        demonstration = stretched_demos.T

        self.Y = np.vstack((self.Y, [demonstration]))  # If necessary to review the demo later?

        # here we concatenate with joint trajectories the final Cartesian position as a context
        if self.with_orientation and self.dist_to_mean(eef_pose[1]) > self.dist_to_mean(-np.array(eef_pose[1])):
            eef_pose = [eef_pose[0], -np.array(eef_pose[1])]
        context = np.hstack(eef_pose) if self.with_orientation else eef_pose[0]
        assert len(context) == self.context_length, \
            "The provided context (eef pose) has {} dims while {} have been declared".format(len(context), self.context_length)
        self.contexts.append(np.array(context).reshape((1, self.context_length)))

        mppi_demo = np.reshape(np.dot(self.MPPI, demonstration).T, (1, self.num_joints * self.num_basis))
        demo_and_context = np.hstack((mppi_demo, self.contexts[-1]))

        # here we flatten all joint trajectories
        self.W_full = np.vstack((self.W_full, demo_and_context))
        self.mean_W_full = np.mean(self.W_full, axis=0)
        if self.num_demos > 1:
            self.cov_W_full = np.cov(self.W_full.T, bias=0)

        self.plot_cartesian_step(eef_pose, False, 'demo_{}'.format(self.num_demos - 1))
        self.plot_joints_step('mean_std_demo_{}'.format(self.num_demos - 1))

    def generate_trajectory(self, cartesian_goal, joint_goal_plot=None, stamp=''):
        """
        Generate a joint trajectory to the given cartesian goal
        :param cartesian_goal: [[x, y, z], [x, y, z, w]]
        :param joint_goal_plot: [j1, j2, ... jn] joint values of the goal for plotting/debugging only
        :return:
        """
        if self.with_orientation and self.dist_to_mean(cartesian_goal[1]) > self.dist_to_mean(-np.array(cartesian_goal[1])):
            cartesian_goal = [cartesian_goal[0], -np.array(cartesian_goal[1])]
        meanNew, CovNew = self.gaussian_conditioning_joints(cartesian_goal)
        output = []
        for joint in range(self.num_joints):
            output.append(np.dot(self.Gn, meanNew[joint*self.num_basis:(joint + 1) * self.num_basis]))
        output = np.array(output).T

        if self.plots != '':
            self.plot_cartesian_step(cartesian_goal, True, stamp)
            joint_goal = joint_goal_plot if joint_goal_plot is not None else output[-1, :]

            # Get mean and std goal
            varNew = np.diagonal(CovNew)
            var_goal, mean_goal = [], []
            for joint in range(self.num_joints):
                var_goal.append(np.dot(self.Gn, varNew[joint*self.num_basis:(joint + 1) * self.num_basis]))
                mean_goal.append(np.dot(self.Gn, meanNew[joint*self.num_basis:(joint + 1) * self.num_basis]))
            self.plot_conditioned_joints_goal(joint_goal, output, mean_goal, var_goal, '_'.join(['end_conditioning', stamp]))
        return output

    @property
    def num_joints(self):
        return self.Y.shape[2]

    @property
    def num_demos(self):
        return self.Y.shape[0]

    @property
    def num_points(self):
        return self.Y.shape[1]

    @property
    def num_viapoints(self):
        return 0 if self.goal is None else 1

    def plot_cartesian_step(self, eef, is_goal=False, stamp=''):
        if self.plots == '':
            return

        mean = self.get_mean_context()
        std = self.get_std_context()

        f = plt.figure(facecolor="white", figsize=(16, 12))

        # Position
        ax = f.add_subplot(121) if self.with_orientation else f.add_subplot(111)
        ax.set_title('End effector position (x, y, z)')

        for dim in range(3):
            ax.errorbar(dim, mean[dim], self.std_factor*std[dim], color=self.colors[dim % len(self.colors)], elinewidth=20)
            ax.plot(dim, eef[0][dim], marker='o', markerfacecolor='red', markersize=10)
            for point in self.plotted_points:
                ax.plot(dim, point[0][dim], marker='o', markerfacecolor='black', markersize=7)
        ax.set_ylim([-1, 1])

        # Orientation
        if self.with_orientation:
            ax = f.add_subplot(122)
            ax.set_title('End effector orientation (x, y, z, w)')
            for dim in range(4):
                ax.errorbar(dim, mean[dim+3], self.std_factor * std[dim+3], color=self.colors[dim % len(self.colors)], elinewidth=20)
                ax.plot(dim, eef[1][dim], marker='o', markerfacecolor='red', markersize=10)
                for point in self.plotted_points:
                    ax.plot(dim, point[1][dim], marker='o', markerfacecolor='black', markersize=7)
            ax.set_ylim([-3.15, 3.15])

        # Save plots
        self._mk_dirs()
        filename = '_'.join(['cartesian', stamp])
        plt.savefig(join(self.plots, filename) + '.svg', dpi=100, transparent=False)
        if not is_goal:
            self.plotted_points.append(eef)
        plt.close('all')

    def plot_conditioned_joints_goal(self, goal, obtained_traj, mean_goal, var_goal, stamp):
        if self.plots == '':
            return

        self._mk_dirs()
        color_id = 0
        mean_joints = self.get_mean_joints()
        std_joints = self.get_std_joints()
        for joint_id, joint in enumerate(goal):
            f = plt.figure(facecolor="white", figsize=(16, 12))
            ax = f.add_subplot(111)
            ax.set_title('Conditioning joint {}: mean, {}std, var(goal), output, goal'.format(joint_id, self.std_factor))
            plt.plot(self.x, mean_joints[joint_id], label='Joint {}'.format(joint_id), color=self.colors[color_id], linestyle='dashed')
            plt.fill_between(self.x, mean_joints[joint_id] - self.std_factor*std_joints[joint_id],
                             mean_joints[joint_id] + self.std_factor*std_joints[joint_id],
                             alpha=0.1, color=self.colors[color_id])
            plt.plot(self.x, mean_goal[joint_id], color=self.colors[color_id])
            plt.fill_between(self.x, mean_goal[joint_id] - var_goal[joint_id],
                             mean_goal[joint_id] + var_goal[joint_id],
                             alpha=0.1, color=self.colors[color_id])
            plt.plot([1], [joint], marker='o', markerfacecolor=self.colors[color_id], markersize=7)
            plt.plot([1], [obtained_traj[-1, joint_id]], marker='o', markerfacecolor=self.colors[color_id], markersize=4)
            plt.plot(self.x, obtained_traj[:, joint_id], color=self.colors[color_id])
            color_id = (color_id + 1) % len(self.colors)
            end_stamp = '_'.join([stamp, 'joint', str(joint_id)])
            plt.savefig(join(self.plots, end_stamp) + '.svg', dpi=100, transparent=False)
            plt.close('all')

    def plot_joints_step(self, stamp):
        if self.plots == '':
            return

        mean_joints = self.get_mean_joints()
        std_joints = self.get_std_joints()
        f = plt.figure(facecolor="white", figsize=(16, 12))
        ax = f.add_subplot(111)
        ax.set_title('Mean +- {}std'.format(self.std_factor))
        color_id = 0
        for joint_id, joint_mean in enumerate(mean_joints):
            ax.plot(self.x, joint_mean, label='Joint {}'.format(joint_id), color=self.colors[color_id], linestyle='dashed')
            plt.fill_between(self.x, joint_mean - self.std_factor*std_joints[joint_id],
                             joint_mean + self.std_factor*std_joints[joint_id],
                             alpha=0.1, color=self.colors[color_id])
            color_id = (color_id + 1) % len(self.colors)
        plt.legend(loc='upper left')
        self._mk_dirs()
        filename = '_'.join(['joints', stamp])
        plt.savefig(join(self.plots, filename) + '.svg', dpi=100, transparent=False)
        plt.close('all')

    def plot_demos(self):
        if self.plots == '':
            return
        yt = self.Y.transpose(2, 0, 1)
        for joint_id, joint in enumerate(yt):
            f = plt.figure(facecolor="white", figsize=(16, 12))
            ax = f.add_subplot(111)
            ax.set_title('Joint {}'.format(joint_id))
            for demo_id, demo in enumerate(joint):
                ax.plot(self.x, demo, label='Demo {}'.format(demo_id))
            plt.legend()
            # Save or show plots
            self._mk_dirs()
            filename = 'demos_of_joint_{}'.format(joint_id)
            plt.savefig(join(self.plots, filename) + '.svg', dpi=100, transparent=False)
            plt.close('all')

    def _mk_dirs(self):
        if not exists(self.plots):
            makedirs(self.plots)
