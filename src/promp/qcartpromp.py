import numpy as np
from scipy.interpolate import interp1d

class QCartProMP(object):
    """
    n-dimensional probabilistic MP storing joints (Q) and end effector (Cart)
    """
    def __init__(self, num_joints=7, num_basis=20, sigma=0.05, num_samples=100, context_length=3):
        self.nrBasis = num_basis
        self.nrTraj = num_samples
        self.context_length = context_length

        self.mu = np.linspace(0, 1, self.nrBasis)
        self.z = np.linspace(0, 1, self.nrTraj).reshape((self.nrTraj, 1))
        self.sigma = sigma * np.ones(self.nrBasis)
        self.Gn = self._generate_basis_fx(self.z, self.mu, self.sigma)

        self.my_linRegRidgeFactor = 1e-8 * np.ones((self.nrBasis, self.nrBasis))
        self.MPPI = np.dot(np.linalg.inv(np.dot(self.Gn.T, self.Gn) + self.my_linRegRidgeFactor), self.Gn.T)
        self.Y = np.empty((0, num_samples, num_joints), float)  # all demonstrations
        self.x = np.linspace(0, 1, num_samples)

        self.W = np.empty((self.num_joints, 0, self.nrBasis), float)
        self.W_full = np.reshape(self.W, (0, self.num_joints * self.nrBasis))
        self.contexts = []
        self.goal = None

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

    @staticmethod
    def _gaussian_conditioning(W_full, obs, noise=.01**2):
        # This is an alternative way to condition the ProMP.
        # It is simple compared to the original ProMP because it does not introduce
        # the features. This is possible because we are conditioning the ProMP on
        # the context, which in this case does not have a feature (feature is
        # identity). This function should not be used if you want to implement the
        # "real" ProMP.
        mean_W_full = np.mean(W_full, axis=0)
        cov_W_full = np.cov(W_full.T)

        # total dimension of the concatenated weight vector, which here also contains the context.
        nDim = len(mean_W_full)

        # number of variables to infer are the weights of the trajectories minus the number of context variables
        q = nDim - obs.shape[0]

        model = {}
        model["Cov11"] = cov_W_full[:q, :q]
        model["Cov12"] = cov_W_full[:q, q:]
        model["Cov21"] = cov_W_full[q:, :q]
        model["Cov22"] = cov_W_full[q:, q:]

        mean_wNew = mean_W_full[:q] + np.dot(model["Cov12"], np.dot(
            np.linalg.inv(model["Cov22"] + noise * np.eye(model["Cov22"].shape[0])), obs - mean_W_full[q:]))
        Cov_wNew = model["Cov11"] - np.dot(model["Cov12"], np.dot(
            np.linalg.inv(model["Cov22"] + noise * np.eye(model["Cov22"].shape[0])), model["Cov21"]))

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
        mppi_demo = np.reshape(np.dot(self.MPPI, demonstration).T, (self.num_joints, 1, self.nrBasis))
        self.W = np.hstack((self.W, mppi_demo))

        # here we flatten all joint trajectories
        self.W_full = self.W.reshape(self.num_demos, self.num_joints * self.nrBasis)

        # here we concatenate with joint trajectories the final Cartesian position as a context
        context = eef_pose[0]  # TODO: only position?
        assert len(context) == self.context_length, \
            "The provided context (eef pose) has {} dims while {} have been declared".format(len(context), self.context_length)
        self.contexts.append(context)
        self.W_full = np.hstack((self.W_full, self.contexts))


    def set_goal(self, obsy, sigmay=1e-6):
        """
        Set goal in task space
        :param obsy: [[x, y, z], [x, y, z, w]]
        :param sigmay: standard deviation TODO
        :return:
        """
        obsy = obsy[0]  # TODO: only position?
        self.goal = np.array(obsy)

    def generate_trajectory(self, randomness=1e-10):
        meanNew, CovNew = self._gaussian_conditioning(self.W_full, self.goal)
        output = []
        for joint in range(self.num_joints):
            output.append(np.dot(self.Gn, meanNew[joint*self.nrBasis:(joint+1)*self.nrBasis]))
        return np.array(output).T

    def clear_viapoints(self):
        raise NotImplementedError("No viapoint except goals atm")

    def add_viapoint(self, t, obsy, sigmay=1e-6):
        raise NotImplementedError("No viapoint except goals atm")

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
