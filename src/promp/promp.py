import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class NDProMP(object):
    """
    n-dimensional ProMP
    """
    def __init__(self, num_joints, nrBasis=11, sigma=0.05, num_samples=100):
        """

        :param num_joints: Number of underlying ProMPs
        :param nrBasis:
        :param sigma:
        """
        if num_joints < 1:
            raise ValueError("You must declare at least 1 joint in a NDProMP")
        self._num_joints = num_joints
        self.promps = [ProMP(nrBasis, sigma, num_samples) for joint in range(num_joints)]
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'chocolate', 'deepskyblue', 'sage', 'darkviolet', 'crimson']

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

    @property
    def num_viapoints(self):
        return self.promps[0].num_viapoints

    @property
    def goal_bounds(self):
        return [joint.goal_bounds for joint in self.promps]

    def get_bounds(self, t):
        """
        Return the bounds of all joints at time t
        :param t: 0 <= t <= 1
        :return: [(lower boundary joints 0, upper boundary joints 0), (lower boundary joint 1), upper)...]
        """
        return [joint.get_bounds(t) for joint in self.promps]

    def clear_viapoints(self):
        for promp in self.promps:
            promp.clear_viapoints()

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

    def generate_trajectory(self, randomness=1e-10):
        trajectory = []
        for joint_demo in range(self.num_joints):
            trajectory.append(self.promps[joint_demo].generate_trajectory(randomness))
        return np.array(trajectory).T[0]

    def plot(self, x=None, joint_names=(), output_randomess=0.5):
        """
        Plot the means and variances of gaussians, requested viapoints as well as an output trajectory (dotted)
        :param output_randomess: 0. to 1., -1 to disable output plotting
        """
        if output_randomess >= 0:
            output = self.generate_trajectory(output_randomess).T

        for promp_idx, promp in enumerate(self.promps):
            color = self.colors[promp_idx % len(self.colors)]
            joint_name = "Joint {}".format(promp_idx+1) if len(joint_names) == 0 else joint_names[promp_idx]
            promp.plot(x, joint_name, color)
            if output_randomess >= 0:
                plt.plot(x, output[promp_idx], linestyle='--', label="Out {}".format(joint_name), color=color, lw=2)


class ProMP(object):
    """
    Uni-dimensional probabilistic MP
    """
    def __init__(self, nrBasis=11, sigma=0.05, num_samples=100):
        self.x = np.linspace(0, 1, num_samples)
        self.nrSamples = len(self.x)
        self.nrBasis = nrBasis
        self.sigma = sigma
        self.sigmaSignal = float('inf')  # Noise of signal (float)
        self.C = np.arange(0,nrBasis)/(nrBasis-1.0)
        self.Phi = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(self.x, (self.nrBasis, 1)).T)).T ** 2 / (self.sigma ** 2)))
        self.Phi /= sum(self.Phi)

        self.viapoints = []
        self.W = np.array([])
        self.nrTraj = 0
        self.meanW = None
        self.sigmaW = None
        self.Y = np.empty((0, self.nrSamples), float)

    def add_demonstration(self, demonstration):
        interpolate = interp1d(np.linspace(0, 1, len(demonstration)), demonstration, kind='cubic')
        stretched_demo = interpolate(self.x)
        self.Y = np.vstack((self.Y, stretched_demo))
        self.nrTraj = len(self.Y)
        self.W = np.dot(np.linalg.inv(np.dot(self.Phi, self.Phi.T)), np.dot(self.Phi, self.Y.T)).T  # weights for each trajectory
        self.meanW = np.mean(self.W, 0)                                                             # mean of weights
        w1 = np.array(map(lambda x: x - self.meanW.T, self.W))
        self.sigmaW = np.dot(w1.T, w1)/self.nrTraj                                                  # covariance of weights
        self.sigmaSignal = np.sum(np.sum((np.dot(self.W, self.Phi) - self.Y) ** 2)) / (self.nrTraj * self.nrSamples)

    @property
    def noise(self):
        return self.sigmaSignal

    @property
    def num_demos(self):
        return self.Y.shape[0]

    @property
    def num_points(self):
        return self.Y.shape[1]

    @property
    def num_viapoints(self):
        return len(self.viapoints)

    @property
    def goal_bounds(self):
        """
        Joint boundaries of the last point
        :return: (lower boundary, upper boundary)
        """
        return self._get_bounds(-1)

    def get_bounds(self, t):
        """
        Return the bounds at time t
        :param t: 0 <= t <= 1
        :return: (lower boundary, upper boundary)
        """
        return self._get_bounds(int(self.num_points*t))

    def _get_bounds(self, t_index):
        mean = np.dot(self.Phi.T, self.meanW)
        std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW, self.Phi))))
        return (mean - std)[t_index], (mean + std)[-1]

    def clear_viapoints(self):
        del self.viapoints[:]

    def add_viapoint(self, t, obsy, sigmay=1e-6):
        """
        Add a viapoint to the trajectory
        Observations and corresponding basis activations
        :param t: timestamp of viapoint
        :param obsy: observed value at time t
        :param sigmay: observation variance (constraint strength)
        :return:
        """
        self.viapoints.append({"t": t, "obsy": obsy, "sigmay": sigmay})

    def set_goal(self, obsy, sigmay=1e-6):
        self.add_viapoint(1., obsy, sigmay)

    def set_start(self, obsy, sigmay=1e-6):
        self.add_viapoint(0., obsy, sigmay)

    def generate_trajectory(self, randomness=1e-10):
        """
        Outputs a trajectory
        :param randomness: float between 0. (output will be the mean of gaussians) and 1. (fully randomized inside the variance)
        :return: a 1-D vector of the generated points
        """
        newMu = self.meanW
        newSigma = self.sigmaW

        for viapoint in self.viapoints:
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(viapoint['t'], (11, 1)).T)).T ** 2 / (self.sigma ** 2)))
            PhiT = PhiT / sum(PhiT)  # basis functions at observed time points

            # Conditioning
            aux = viapoint['sigmay'] + np.dot(np.dot(PhiT.T, newSigma), PhiT)
            newMu = newMu + np.dot(np.dot(newSigma, PhiT) * 1 / aux, (viapoint['obsy'] - np.dot(PhiT.T, newMu)))  # new weight mean conditioned on observations
            newSigma = newSigma - np.dot(np.dot(newSigma, PhiT) * 1 / aux, np.dot(PhiT.T, newSigma))

        sampW = np.random.multivariate_normal(newMu, randomness*newSigma, 1).T
        return np.dot(self.Phi.T, sampW)

    def plot(self, x=None, legend='promp', color=None):
        mean = np.dot(self.Phi.T, self.meanW)
        x = self.x if x is None else x
        plt.plot(x, mean, color=color, label=legend)
        std = 2*np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW, self.Phi))))
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        for viapoint_id, viapoint in enumerate(self.viapoints):
            x_index = x[int(round((len(x)-1)*viapoint['t'], 0))]
            plt.plot(x_index, viapoint['obsy'], marker="o", markersize=10, label="Via {} {}".format(viapoint_id, legend), color=color)
