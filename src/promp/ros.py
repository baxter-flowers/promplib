from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory, RobotState
from numpy import mean, array
from rospy import Duration
from matplotlib.pyplot import show
from .promp import NDProMP


class ProMP(object):
    def __init__(self, num_joints=7):
        self._num_joints = num_joints
        self._durations = []
        self.promp = NDProMP(num_joints)
        self.joint_names = []

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
            raise TypeError("ros.ProMP.add_demonstration only accepts RT or JT, got {}".format(type(demonstration)))

        if len(self.joint_names) > 0 and self.joint_names != demonstration.joint_names:
            raise ValueError("Joints must be the same and in same order for all demonstrations, this demonstration has joints {} while we had {}".format(demonstration.joint_names, self.joint_names))

        self._durations.append(demonstration.points[-1].time_from_start.to_sec() - demonstration.points[0].time_from_start.to_sec())
        self.joint_names = demonstration.joint_names
        demo_array = [jtp.positions for jtp in demonstration.points]
        self.promp.add_demonstration(demo_array)

    @property
    def num_demos(self):
        return self.promp.num_demos

    @property
    def num_points(self):
        return self.promp.num_points

    @property
    def mean_duration(self):
        return mean(self._durations)

    def add_viapoint(self, t, obsys, sigmay=1e-6):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: Time of observation
        :param obsys: RobotState observed at the time
        :param sigmay:
        :return:
        """
        if isinstance(obsys, RobotState):
            obsys = obsys.joint_state
        elif not isinstance(obsys, JointState):
            raise TypeError("ros.ProMP.add_viapoint only accepts RS or JS, got {}".format(type(obsys)))
        self.promp.add_viapoint(t, map(float, obsys.position), sigmay)

    def set_goal(self, obsy, sigmay=1e-6):
        if isinstance(obsy, RobotState):
            obsy = obsy.joint_state
        elif not isinstance(obsy, JointState):
            raise TypeError("ros.ProMP.set_goal only accepts RS or JS, got {}".format(type(obsy)))
        self.promp.set_goal(map(float, obsy.position), sigmay)

    def set_start(self, obsy, sigmay=1e-6):
        if isinstance(obsy, RobotState):
            obsy = obsy.joint_state
        elif not isinstance(obsy, JointState):
            raise TypeError("ros.ProMP.set_start only accepts RS or JS, got {}".format(type(obsy)))
        self.promp.set_start(map(float, obsy.position), sigmay)

    def generate_trajectory(self, randomness=True, duration=-1):
        """
        Generate a new trajectory from the given demonstrations and parameters
        :param randomness: True if the output trajectory must be randomised
        :param duration: Desired duration, auto if duration < 0
        :return: the generated RobotTrajectory message
        """
        trajectory_array = self.promp.generate_trajectory(randomness)
        trajectory_array = array(trajectory_array).T
        rt = RobotTrajectory()
        rt.joint_trajectory.joint_names = self.joint_names
        duration = float(self.mean_duration) if duration < 0 else duration
        for point_idx, point in enumerate(trajectory_array):
            time = point_idx*duration/float(self.num_points)
            jtp = JointTrajectoryPoint(positions=map(float, point), time_from_start=Duration(time))
            rt.joint_trajectory.points.append(jtp)
        return rt

    def plot(self, randomness=True):
        self.promp.plot(randomness, self.joint_names)
        show()
