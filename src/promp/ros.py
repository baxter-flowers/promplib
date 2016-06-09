from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory, RobotState
from nav_msgs.msg import Path
from numpy import mean, linspace
from rospy import Duration
from matplotlib.pyplot import show, legend
from transformations import pose_to_list, list_to_raw_list, raw_list_to_list
from .promp import NDProMP
from .ik import IK as _IK


class IK(object):
    def __init__(self, arm, k=2):
        self._ik = _IK(arm, k)

    def get(self, x_des, seed=None, bounds=()):
        """
        Get the IK by minimization
        :param x_des: desired task space pose [[x, y, z], [x, y, z, w]]
        :param seed: RobotState message
        :param bounds: promp mean-std, mean+std
        :return: (bool, joints)
        """
        if isinstance(seed, RobotState):
            seed = seed.joint_state
        elif not isinstance(seed, JointState) and seed is not None:
            raise TypeError('ros.IK.get only accepts RS or JS, got {}'.format(type(seed)))

        seed = [seed.position[seed.name.index(joint)] for joint in self._ik.joints] if seed is not None else ()
        result = self._ik.get(x_des, seed, bounds)
        return result[0], JointState(name=self._ik.joints, position=list(result[1]))

    def get_multiple(self, x_des_list, seed=None, bounds=()):
        """
        Get multiple IKs whose points follow each other
        :param x_des_list: The list of end effector
        :param seed: RobotState message
        :param bounds:
        :return: [(bool, joints), (bool, joints), ...]
        """
        if not isinstance(x_des_list, list) and not isinstance(x_des_list, tuple):
            raise TypeError('ros.IK.get_multiple only accepts lists, got {}'.format(type(x_des_list)))

        def get_seed(output):
            # Find the last valid seed
            for point in range(-1, -len(output)-1, -1):
                if output[point][0]:
                    return output[point][1]
            return seed

        output = []
        for x_des in x_des_list:
            selected_seed = get_seed(output)
            output.append(self.get(x_des, selected_seed, bounds))
        return output


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
    def num_viapoints(self):
        return self.promp.num_viapoints

    @property
    def mean_duration(self):
        return float(mean(self._durations))

    def clear_viapoints(self):
        self.promp.clear_viapoints()

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
        try:
            positions = [obsys.position[obsys.name.index(joint)] for joint in self.joint_names]  # Make sure joints are in right order
        except KeyError as e:
            raise KeyError("Joint {} provided as viapoint is unknown to the demonstrations".format(e))
        else:
            self.promp.add_viapoint(t, map(float, positions), sigmay)

    def set_goal(self, obsy, sigmay=1e-6):
        if isinstance(obsy, RobotState):
            obsy = obsy.joint_state
        elif not isinstance(obsy, JointState):
            raise TypeError("ros.ProMP.set_goal only accepts RS or JS, got {}".format(type(obsy)))
        try:
            positions = [obsy.position[obsy.name.index(joint)] for joint in self.joint_names]  # Make sure joints are in right order
        except KeyError as e:
            raise KeyError("Joint {} provided as goal state is unknown to the demonstrations".format(e))
        else:
            self.promp.set_goal(map(float, positions), sigmay)

    def set_start(self, obsy, sigmay=1e-6):
        if isinstance(obsy, RobotState):
            obsy = obsy.joint_state
        elif not isinstance(obsy, JointState):
            raise TypeError("ros.ProMP.set_start only accepts RS or JS, got {}".format(type(obsy)))
        try:
            positions = [obsy.position[obsy.name.index(joint)] for joint in self.joint_names]  # Make sure joints are in right order
        except KeyError as e:
            raise KeyError("Joint {} provided as start state is unknown to the demonstrations".format(e))
        else:
            self.promp.set_start(map(float, positions), sigmay)

    def generate_trajectory(self, randomness=1e-10, duration=-1):
        """
        Generate a new trajectory from the given demonstrations and parameters
        :param duration: Desired duration, auto if duration < 0
        :return: the generated RobotTrajectory message
        """
        trajectory_array = self.promp.generate_trajectory(randomness)
        rt = RobotTrajectory()
        rt.joint_trajectory.joint_names = self.joint_names
        duration = float(self.mean_duration) if duration < 0 else duration
        for point_idx, point in enumerate(trajectory_array):
            time = point_idx*duration/float(self.num_points)
            jtp = JointTrajectoryPoint(positions=map(float, point), time_from_start=Duration(time))
            rt.joint_trajectory.points.append(jtp)
        return rt

    def plot(self, output_randomess=0.5):
        """
        Plot the means and variances of gaussians, requested viapoints as well as an output trajectory (dotted)
        :param output_randomess: 0. to 1., -1 to disable output plotting
        """
        self.promp.plot(linspace(0, self.mean_duration, self.num_points), self.joint_names, output_randomess)
        legend(loc="upper left")
        show()
