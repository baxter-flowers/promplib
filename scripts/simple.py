import promp
import numpy as np
import matplotlib.pyplot as plt

p = promp.ProMP()

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrSamples=len(x);                    # number of time points in trajectory
nrTraj=30                            # number of trajectoreis for training
sigmaNoise=0.02                      # noise on training trajectories
A = np.array([.2, .2, .01, -.05])
X = np.vstack( (np.sin(5*x), x**2, x, np.ones((1,len(x))) ))

Y = np.zeros( (nrTraj,len(x)) )
for traj in range(0, nrTraj):
    sample = np.dot(A + sigmaNoise * np.random.randn(1,4), X)[0]
    label = 'training' if traj==0 else ''
    plt.plot(x, sample, 'b', label=label)
    p.add_demonstration(sample)

p.add_viapoint(0.7, 5)
#p.set_goal(-5)

for i in np.arange(0,10):
    label = 'output' if i==0 else ''
    plt.plot(x, p.generate_trajectory(), 'r', label=label)

plt.legend()
plt.show()
