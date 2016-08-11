import numpy as np
from scipy.interpolate import interp1d

class QCartProMP(object):
    """
    n-dimensional probabilistic MP storing joints (Q) and end effector (Cart)
    """
    def __init__(self, num_joints=7, num_basis=20, sigma=0.05, noise=.01, num_samples=100, with_orientation=True):
        self.num_basis = num_basis
        self.nrTraj = num_samples
        self.with_orientation = with_orientation
        self.context_length = 7 if with_orientation else 3

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

    def _gaussian_conditioning(self, obs):
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

        mean_wNew = self.mean_W_full[:self.nQ] + np.dot(model["Cov12"], np.dot(
            np.linalg.inv(model["Cov22"] + self.noise * np.eye(model["Cov22"].shape[0])), obs - self.mean_W_full[self.nQ:]))
        Cov_wNew = model["Cov11"] - np.dot(model["Cov12"], np.dot(
            np.linalg.inv(model["Cov22"] + self.noise * np.eye(model["Cov22"].shape[0])), model["Cov21"]))

        return mean_wNew, Cov_wNew

    def gaussian_conditioning_context(self, goal):
        """
        Condition a temporary goal (not set) and return the mean and covariance of its context
        :param goal: [[x, y, z], [x, y, z, w]]
        :return: (mean, cov) of dimensions 3 and 3, 3 or 7 and 7, 7 if with_orientation
        """
        meanNew, CovNew = self._gaussian_conditioning(goal)
        return meanNew[self.nQ:], CovNew[self.nQ:, self.nQ:]

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
        context = eef_pose[0] + eef_pose[1] if self.with_orientation else eef_pose[0]
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

    def set_goal(self, obsy):
        """
        Set goal in task space
        :param obsy: [[x, y, z], [x, y, z, w]]
        :return:
        """
        obsy = obsy[0] + obsy[1] if self.with_orientation else obsy[0]
        self.goal = np.array(obsy)

    def generate_trajectory(self):
        """
        Generate a joint trajectory when a goal has preliminarily been set
        :return:
        """
        if self.goal is None:
            raise RuntimeError("Set a goal before generating a trajectory")

        meanNew, CovNew = self._gaussian_conditioning(self.goal)
        output = []
        for joint in range(self.num_joints):
            output.append(np.dot(self.Gn, meanNew[joint*self.num_basis:(joint + 1) * self.num_basis]))
        return np.array(output).T

    def clear_viapoints(self):
        self.goal = None

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

