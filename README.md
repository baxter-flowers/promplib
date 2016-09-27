# Vocal interactive learning of Clustered Probabilistic Movement Primitives

Python implementation of [Probabilistic Movement Primitives (ProMP)](https://papers.nips.cc/paper/5177-probabilistic-movement-primitives.pdf) with an optional ROS overlay.

## Joint-space primitives with a task-space constraint
The primitives are stored in joint-space but demonstrations are provided both in joint space and task space, context. Thanks to this context, task-space goals can be requested to these joint-space primitives. The benefit is that requesting a new task-space goal does not require to call an IK method which would return demonstrations-agnostic joint configurations. 

## Vocal interactive learning and clustering
This work includes an interactive learning aspect which allows to automatically cluster motor primitives based on the standard deviation of their demonstrations. A new primitive is created automatically if the provided demonstration is out of 2 standard deviation of the existing primitives, otherwise the demonstration is distributed to an existing one.

## Classes
![Class diagram](http://yuml.me/diagram/class/[QCartProMP]%3C0..*-++[InteractiveProMP],%20[ReplayableInteractiveProMP]-%5E[InteractiveProMP],%20[ROSQCartProMP]-%5E[QCartProMP],%20[ROSInteractiveProMP]-%5E[InteractiveProMP],%20[ROSReplayableInteractiveProMP]-%5E[ReplayableInteractiveProMP])

 - `QCartProMP` is a "Q + Cartesian" probabilistic movement primitive, i.e. the mother class representing a joint-space primitive including a cartesian context (position + orientation).
 - `InteractiveQCartProMP` owns several `QCartProMP`, each corresponding to a cluster.
 - `ReplayabaleInteractiveQCartProMP` has the same interface than `InteractiveQCartProMP` except that it also dumps all actions (add a new demo or set a new goal) in files for further perfect replays of the sequence of actions.
 - Each of these classes has inputs and outputs in the form of Python lists, but has also a ROS overlay providing the same methods using ROS messages instead (mainly `JointTrajectory` and `JointState`).

In general it is a good idea to always use the highest level (`ReplayableInteractiveProMP`) to be able to reproduce and compare results, but (`InteractiveProMP`) has the same behaviour and API without file persistence, recorded primitives die at program closure.

## Usage
### Direct usage with Baxter
This package works out-of-the-box on ROS + Baxter with the following dependencies installed:
 - numpy, scipy (`pip install numpy scipy`)
 - ROS package [baxter_commander](https://github.com/baxter-flowers/baxter_commander/)
 - ROS package [baxter_pykdl](https://github.com/RethinkRobotics/baxter_pykdl)
 - C# and ROS packages [Kinect 2 server and Python client](https://github.com/baxter-flowers/kinect_2_server/) (only used with vocal interaction, with the server running on an extra Windows 10 machine)
 
### Use a different robot
Using a different robot only requires replacing the forward kinematics -and optionnally inverse kinematics- class(es) to produce answers accordingly to your robot. The only constraint is that [the constructors](src/promp/ik.py#L48), [`get(...)`](src/promp/ik.py#L52), and [`joints`](src/promp/ik.py#L62) exist and keep the same signature.

Since this package is both non and non-ROS compatible, standard ROS messages for FK and IK are not used, so you must replace or overload these classes by respecting the format of the inputs/outputs described in the code.

The provided IK is optimization-based, you may keep this implementation and override only the FK or replace both.

### Execute
Demonstrations are recorded and clustered in the mean time, through one of these methods:
 - The ipython notebook [test_replayable](notebooks/test_replayable.ipynb) (`recording` section); or
 - The script [Vocal interactive learning](scripts/vocal_interactive_promps.py), which has the vocal interaction aspect, by saying `record a motion`

Movements can be computed and played, through one of these methods:
 - The ipython notebook [test_replayable](notebooks/test_replayable.ipynb) (`reading` section); or
 - The script [Vocal interactive learning](scripts/vocal_interactive_promps.py), which has the vocal interaction aspect, by saying `set a goal`
 - The script [Replay](scripts/replay.py) which replays the exact same sequence of demonstrations and tentatives of goal reaching
