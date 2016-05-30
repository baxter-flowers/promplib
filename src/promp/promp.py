import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class NDProMP(object):
    """
    n-dimensional ProMP
    """
    def __init__(self, num_joints, nrBasis=11, sigma=0.05, num_samples=500):
        """

        :param num_joints: Number of underlying ProMPs
        :param nrBasis:
        :param sigma:
        """
        if num_joints < 1:
            raise ValueError("You must declare at least 1 joint in a NDProMP")
        self._num_joints = num_joints
        self.promps = [ProMP(nrBasis, sigma, num_samples) for joint in range(num_joints)]

    @property
    def num_joints(self):
        return self._num_joints

    @property
    def x(self):
        return self.promps[0].x

    def add_demonstration(self, demonstration):
        """
        Add a new N-joints demonstration[time][joint] and update the model
        :param demonstration: List of "num_joints" demonstrations
        :return:
        """
        demonstration = np.array(demonstration).T  # Revert the representation for each time for each joint, for each joint for each time

        if len(demonstration) != self.num_joints:
            raise ValueError("The given demonstration has {} joints while num_joints={}".format(len(demonstration), self.num_joints))

        for joint_demo_idx, joint_demo in enumerate(demonstration):
            self.promps[joint_demo_idx].add_demonstration(joint_demo)

    @property
    def num_demos(self):
        return self.promps[0].num_demos

    @property
    def num_points(self):
        return self.promps[0].num_points

    def add_viapoint(self, t, obsys, sigmay=1e-6):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: Time of observation
        :param obsys: List of observations obys[joint] for each joint
        :param sigmay:
        :return:
        """
        if len(obsys) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(obsys), self.num_joints))

        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].add_viapoint(t, obsys[joint_demo], sigmay)

    def set_goal(self, obsy, sigmay=1e-6):
        if len(obsy) != self.num_joints:
            raise ValueError("The given goal state has {} joints while num_joints={}".format(len(obsy), self.num_joints))

        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].set_goal(obsy[joint_demo], sigmay)

    def set_start(self, obsy, sigmay=1e-6):
        if len(obsy) != self.num_joints:
            raise ValueError("The given start state has {} joints while num_joints={}".format(len(obsy), self.num_joints))

        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].set_start(obsy[joint_demo], sigmay)

    def generate_trajectory(self, randomness=True):
        trajectory = []
        for joint_demo in range(self.num_joints):
            trajectory.append(self.promps[joint_demo].generate_trajectory(randomness))
        return trajectory

    def plot(self, x=None, joint_names=()):
        for promp_idx, promp in enumerate(self.promps):
            promp.plot(x, "Joint {}".format(promp_idx+1) if len(joint_names) == 0 else joint_names[promp_idx])


class ProMP(object):
    """
    Uni-dimensional probabilistic MP
    """
    def __init__(self, nrBasis=11, sigma=0.05, num_samples=500):
        self.x = np.linspace(0, 1, num_samples)
        self.nrSamples = len(self.x)
        self.nrBasis = nrBasis
        self.sigma = sigma
        self.nrSamples = len(self.x)
        self.C = np.arange(0,nrBasis)/(nrBasis-1.0)
        self.Phi = np.exp(-.5 * (np.tile(self.x, (nrBasis, 1)).T - self.C) ** 2 / (sigma ** 2))

        self.W = np.array([])
        self.nrTraj = 0
        self.meanW = None
        self.sigmaW = None
        self.Y = np.empty((0, self.nrSamples), float)
        self.newMu = None
        self.newSigma = None

    def add_demonstration(self, demonstration):
        interpolate = interp1d(np.linspace(0, 1, len(demonstration)), demonstration, kind='cubic')
        stretched_demo = interpolate(self.x)
        self.Y = np.vstack((self.Y, stretched_demo))
        self.nrTraj = len(self.Y)
        a = np.linalg.inv(np.dot(self.Phi.T, self.Phi))
        b = np.dot(self.Phi.T, self.Y.T)
        self.W = np.dot(a, b)                                                             # weights for each trajectory
        self.meanW = np.mean(self.W, 1)                                                   # mean of weights
        self.sigmaW = np.dot((self.W.T-self.meanW).T, (self.W.T-self.meanW))/self.nrTraj  # covariance of weights

    @property
    def num_demos(self):
        return self.Y.shape[0]

    @property
    def num_points(self):
        return self.Y.shape[1]

    def add_viapoint(self, t, obsy, sigmay=1e-6):
        """
        Add a viapoint to the trajectory
        Observations and corresponding basis activations
        :param t: timestamp of viapoint
        :param obsy: observed value at time t
        :param sigmay: observation variance (constraint strength)
        :return:
        """
        PhiT = np.exp(-.5*(np.tile(t, (self.nrBasis,1)).T - self.C)**2/(self.sigma**2))
        PhiT = PhiT / sum(PhiT)     # basis functions at observed time points

        # Conditioning
        aux = sigmay + np.dot(np.dot(PhiT, self.sigmaW), PhiT.T)

        self.newMu = self.meanW + np.dot(np.dot(self.sigmaW,PhiT.T) * 1/aux, (obsy - np.dot(PhiT, self.meanW.T)))   # new weight mean conditioned on observations
        self.newSigma = self.sigmaW - np.dot(np.dot(self.sigmaW, PhiT.T) * 1/aux, np.dot(PhiT, self.sigmaW))

    def set_goal(self, obsy, sigmay=1e-6):
        self.add_viapoint(1., obsy, sigmay)

    def set_start(self, obsy, sigmay=1e-6):
        self.add_viapoint(0., obsy, sigmay)

    def generate_trajectory(self, randomness=True):
        if randomness:
            sampW = np.random.multivariate_normal(self.newMu, self.newSigma, 1).T
            return np.dot(self.Phi, sampW)
        else:
            return np.dot(self.Phi, self.meanW)

    def plot(self, x=None, legend='promp'):
        plt.plot(self.x if x is None else x, self.generate_trajectory(), label=legend)
