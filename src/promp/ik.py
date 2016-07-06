from scipy.optimize import minimize
from baxter_pykdl import baxter_kinematics
from numpy.linalg import norm
from numpy import dot, array, pi


class IK(object):
    def __init__(self, arm, k=2):
        self._kdl = baxter_kinematics(arm)
        self._joints = map(lambda j: arm + '_' + j, ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2'])
        self.k = k

    def cost_ik(self, q, x_des):
        q_dict = dict(zip(self._joints, q))
        fk = self._kdl.forward_position_kinematics(q_dict)
        return self.cost_position(x_des, fk) + self.cost_orientation(x_des, fk)*self.k

    @staticmethod
    def cost_position(x_des, fk):
        return norm(fk[:3] - x_des[:3]) ** 2

    @staticmethod
    def cost_orientation(x_des, fk):
        return 1 - dot(x_des[3:], fk[3:]) ** 2

    @property
    def joints(self):
        return self._joints

    def get(self, x_des, seed=(), bounds=()):
        """
        Get the IK by minimization
        :param x_des: desired task space pose [[x, y, z], [x, y, z, w]] or flattened [x, y, z, x, y, z, w]
        :param seed: ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        :param bounds: [(min, max), (min, max), (min, max), ... for each joint]
        :return: (bool, joints)
        """
        if len(bounds) != len(self._joints):
            bounds = [(-pi, pi) for j in self._joints]
        if len(seed) == 0:
            seed = [0. for j in self._joints]
        args = [element for component in x_des for element in component] if len(x_des) == 2 else x_des
        result = minimize(self.cost_ik, seed, args=[args], bounds=bounds, method='L-BFGS-B')
        return result.success, result.x


class FK(object):
    def __init__(self, arm):
        self._kdl = baxter_kinematics(arm)
        self._joints = map(lambda j: arm + '_' + j, ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2'])

    def get(self, joints):
        """
        Get the FK from pykdl
        :param joints: ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        :return: [[x, y, z], [x, y, z, w]]
        """
        fk = self._kdl.forward_position_kinematics(dict(zip(self.joints, joints)))
        return [fk[:3], fk[3:]]

    @property
    def joints(self):
        return self._joints
