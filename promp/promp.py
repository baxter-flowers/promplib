import numpy as np


class ProMP(object):
    def __init__(self, nrBasis=11, sigma=0.05):
        self.x = np.arange(0, 1.01, 0.01)
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
        self.Y = np.vstack((self.Y, demonstration))
        self.nrTraj = len(self.Y)
        a = np.linalg.inv(np.dot(self.Phi.T, self.Phi))
        b = np.dot(self.Phi.T, self.Y.T)
        self.W = np.dot(a, b)                                                             # weights for each trajectory
        self.meanW = np.mean(self.W, 1)                                                   # mean of weights
        self.sigmaW = np.dot((self.W.T-self.meanW).T, (self.W.T-self.meanW))/self.nrTraj  # covariance of weights

    @property
    def num_demos(self):
        return self.Y.shape[0]

    def add_viapoint(self, t, obsy, sigmay=.1**2):
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

    def set_goal(self, obsy, sigmay=.1**2):
        self.add_viapoint(1., obsy, sigmay)

    def set_start(self, obsy, sigmay=.1**2):
        self.add_viapoint(0., obsy, sigmay)

    def generate_trajectory(self, randomness=True):
        if randomness:
            sampW = np.random.multivariate_normal(self.newMu, self.newSigma, 1).T
            return np.dot(self.Phi, sampW)
        else:
            return np.dot(self.Phi, self.meanW)
