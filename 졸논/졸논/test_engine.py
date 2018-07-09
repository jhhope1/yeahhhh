import environment
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()
endtime = 10

def plot_robot(points):
    num_points = np.shape(points)[0]
    x = np.zeros((num_points), dtype=np.float64)
    y = np.zeros((num_points), dtype=np.float64)
    z = np.zeros((num_points), dtype=np.float64)
    for i in range(num_points):
        x[i] = points[i].flatten()[0]
        y[i] = points[i].flatten()[1]
        z[i] = points[i].flatten()[2]
    #ax.scatter(x, y, z)
    ax.plot([x[3], x[15]],[y[3], y[15]],[z[3], z[15]])
    ax.plot([x[15], x[14]],[y[15], y[14]],[z[15], z[14]])
    ax.plot([x[14], x[13]],[y[14], y[13]],[z[14], z[13]])
    ax.plot([x[2], x[12]],[y[2], y[12]],[z[2], z[12]])
    ax.plot([x[12], x[11]],[y[12], y[11]],[z[12], z[11]])
    ax.plot([x[11], x[10]],[y[11], y[10]],[z[11], z[10]])
    ax.plot([x[0], x[6]],[y[0], y[6]],[z[0], z[6]])
    ax.plot([x[6], x[5]],[y[6], y[5]],[z[6], z[5]])
    ax.plot([x[5], x[4]],[y[5], y[4]],[z[5], z[4]])
    ax.plot([x[1], x[9]],[y[1], y[9]],[z[1], z[9]])
    ax.plot([x[9], x[8]],[y[9], y[8]],[z[9], z[8]])
    ax.plot([x[8], x[7]],[y[8], y[7]],[z[8], z[7]])
    ax.plot([x[13], x[10]],[y[13], y[10]],[z[13], z[10]])
    ax.plot([x[10], x[4]],[y[10], y[4]],[z[10], z[4]])
    ax.plot([x[4], x[7]],[y[4], y[7]],[z[4], z[7]])
    ax.plot([x[7], x[13]],[y[7], y[13]],[z[7], z[13]])

    ax.set_zlim(0, 0.5)
    ax.set_xlim(points[0][0]-0.5, points[0][0]+0.5)
    ax.set_ylim(points[0][1]-0.5, points[0][1]+0.5)
    plt.draw()

#if __name__ == "__main__":
def test(actor):
    env = environment.Env()
    state_size = env.state_size
    action_size = env.action_size
    env.reset()
    #  env.R.body.wb = np.array([[1.], [-1.], [1.]])
    #  env.R.body.vs = np.array([[0.], [2.], [1.]])
    for t in range(int(endtime/env.R.dtime)):
        print('t = ', t)
        action = actor(np.reshape(env.state, [1, env.state_size]), batch_size=1)
        #  np.zeros((12), dtype=np.float64)
        next_state, reward, done = env.step(action)
        points = np.concatenate([np.reshape(env.R.body.Rnow, (4, 3)), np.reshape(env.R.joints, (12, 3))], axis=0)
        if (t==0):
            plt.show()
        if (done == 1):
            print(done)
            break
        plot_robot(points)
        plt.pause(0.01)
        ax.clear()