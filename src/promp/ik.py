from scipy.optimize import minimize
from baxter_pykdl import baxter_kinematics
from numpy.linalg import norm
from numpy import dot, array, pi


class IK(object):
    def __init__(self, arm, k=2):
        self._kdl = baxter_kinematics(arm)
        self.joints = map(lambda j: arm + '_' + j, ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2'])
        self.k = k

    def cost_ik(self, q, x_des):
        q_dict = dict(zip(self.joints, q))
        fk = self._kdl.forward_position_kinematics(q_dict)
        return self.cost_position(x_des, fk) + self.cost_orientation(x_des, fk)*self.k

    @staticmethod
    def cost_position(x_des, fk):
        return norm(fk[:3] - x_des[:3]) ** 2

    @staticmethod
    def cost_orientation(x_des, fk):
        return 1 - dot(x_des[3:], fk[3:]) ** 2

    def get(self, x_des, seed=(), bounds=()):
        """
        Get the IK by minimization
        :param x_des: desired task space pose [[x, y, z], [x, y, z, w]] or flattened [x, y, z, x, y, z, w]
        :param seed: ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        :param bounds: promp mean-std, mean+std
        :return: (bool, joints)
        """
        if len(bounds) != len(self.joints):
            bounds = [(-pi, pi) for j in self.joints]
        if len(seed) == 0:
            seed = [0. for j in self.joints]
        args = [array(x_des[0] + x_des[1])] if len(x_des) == 2 else [x_des]
        result = minimize(self.cost_ik, seed, args=args, bounds=bounds, method='L-BFGS-B')
        return result.success, result.x
