import numpy as np
from scipy.interpolate import interp1d
from os.path import join, exists
from os import makedirs
import matplotlib.pyplot as plt
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class QCartProMP(object):
    """
    n-dimensional probabilistic MP storing joints (Q) and end effector (Cart)
    """
    def __init__(self, num_joints=7, num_basis=20, sigma=0.05, noise=.01, num_samples=100, with_orientation=True, std_factor=2):
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

        self.mean_W_full = np.zeros(self.context_length)
        self.cov_W_full = self.noise * np.ones((self.context_length, self.context_length))

        self.plotted_points = []

    def _generate_basis_fx(self, z, mu, sigma):
        # print "z", z
        # print "m", mu
        # print "s", sigma
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
        if eef_pose[1][3] < 0:
            eef_pose_rpy = [eef_pose[0], -np.array(eef_pose[1])]
        else:
            eef_pose_rpy = eef_pose
        context = np.hstack(eef_pose_rpy) if self.with_orientation else eef_pose[0]
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

        self.plot_step(eef_pose_rpy, 'demo_{}'.format(self.num_demos-1), path='/tmp/plots')

    def generate_trajectory(self, goal, path_plot=''):
        """
        Generate a joint trajectory to the given goal
        :param goal: [[x, y, z], [x, y, z, w]]
        :param path_plot:
        :return:
        """
        if goal[1][3] < 0:
            goal_rpy = [goal[0], -np.array(goal[1])] #[goal[0], euler_from_quaternion(goal[1])]
        else:
            goal_rpy = goal
        meanNew, CovNew = self.gaussian_conditioning_joints(goal_rpy)
        output = []
        for joint in range(self.num_joints):
            output.append(np.dot(self.Gn, meanNew[joint*self.num_basis:(joint + 1) * self.num_basis]))
        output = np.array(output).T

        if path_plot != '':
            lines = []
            for joint in range(self.num_joints):
                lines.append(plt.plot(self.x, output))
            plt.legend(['Joint {}'.format(j) for j in range(self.num_joints)], loc='upper left')
            plt.savefig(join(path_plot, 'output') + '.svg', dpi=100, transparent=False)
            plt.close('all')

        self.plot_step(goal_rpy, 'goal', True, path='/tmp/plots')
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

    def plot_step(self, eef, stamp='', is_goal=False, path=''):
        mean = self.get_mean_context()
        std = self.get_std_context()

        f = plt.figure(facecolor="white")

        # Position
        ax = f.add_subplot(121) if self.with_orientation else f.add_subplot(111)
        ax.set_title('End effector position (x, y, z)')

        colors = ['tomato', 'darkseagreen', 'cornflowerblue']
        for dim in range(3):
            ax.errorbar(dim, mean[dim], self.std_factor*std[dim], color=colors[dim % len(colors)], elinewidth=20)
            ax.plot(dim, eef[0][dim], marker='o', markerfacecolor='red', markersize=10)
            for point in self.plotted_points:
                ax.plot(dim, point[0][dim], marker='o', markerfacecolor='black', markersize=7)
        ax.set_ylim([-1, 1])

        # Orientation
        if self.with_orientation:
            ax = f.add_subplot(122)
            ax.set_title('End effector orientation (x, y, z, w)')
            for dim in range(4):
                ax.errorbar(dim, mean[dim+3], self.std_factor * std[dim+3], color=colors[dim % len(colors)], elinewidth=20)
                ax.plot(dim, eef[1][dim], marker='o', markerfacecolor='red', markersize=10)
                for point in self.plotted_points:
                    ax.plot(dim, point[1][dim], marker='o', markerfacecolor='black' if is_goal else 'grey',
                            markersize=7 if is_goal else 5)
            ax.set_ylim([-3.15, 3.15])

        # Save plots
        if path != '':
            if not exists(path):
                makedirs(path)
            filename = '_'.join(['cartesian', stamp])
            self.plotted_points.append(eef)
            plt.savefig(join(path, filename) + '.svg', dpi=100, facecolor=f.get_facecolor(), transparent=False)
        else:
            plt.show()
        plt.close('all')

    def plot_demos(self, path=''):
        yt = self.Y.transpose(2, 0, 1)
        for joint_id, joint in enumerate(yt):
            f = plt.figure(facecolor="white")
            ax = f.add_subplot(111)
            ax.set_title('Joint {}'.format(joint_id))
            for demo_id, demo in enumerate(joint):
                ax.plot(self.x, demo, label='Demo {}'.format(demo_id))
            plt.legend()
            # Save or show plots
            if path != '':
                if not exists(path):
                    makedirs(path)
                filename = 'joint_{}_demos'.format(joint_id)
                f.set_size_inches(12.8, 10.24)
                plt.savefig(join(path, filename) + '.svg', dpi=100, facecolor=f.get_facecolor(), transparent=False)
            else:
                plt.show()
            plt.close('all')
