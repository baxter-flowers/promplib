from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from .promp import NDProMP

class ProMP(object):
    def __init__(self, num_joints=7):
        self._num_joints = num_joints
        self.promp = NDProMP(num_joints)

    @property
    def num_joints(self):
        return self._num_joints

    def add_demonstration(self, demonstration):
        """
        Add a new  demonstration and update the model
        :param demonstration: RobotTrajectory or JointTrajectory object
        :return:
        """
        if isinstance(demonstration, RobotTrajectory):
            demonstration = demonstration.joint_trajectory
        elif not isinstance(demonstration, JointTrajectory):
            raise TypeError("ProMPROS.ProMP.add_demonstration only accepts RT or JT, got {}".format(type(demonstration)))

        demo_array = [jtp.positions for jtp in demonstration.points]
        self.promp.add_demonstration(demo_array)

    @property
    def num_demos(self):
        return self.promp.num_demos

    @property
    def num_points(self):
        return self.promp.num_points

    def add_viapoint(self, t, obsys, sigmay=.1 ** 2):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: Time of observation
        :param obsys: RobotState observed at the time
        :param sigmay:
        :return:
        """
        self.promp.add_viapoint(t, obsys, sigmay)

    def set_goal(self, obsy, sigmay=.1 ** 2):
        self.promp.set_goal(obsy, sigmay)

    def set_start(self, obsy, sigmay=.1 ** 2):
        self.promp.set_start(obsy, sigmay)

    def generate_trajectory(self, randomness=True):
        trajectory = []
        for joint_demo in range(self.num_joints):
            trajectory.append(self.promps[joint_demo].generate_trajectory(randomness))
        return trajectory
