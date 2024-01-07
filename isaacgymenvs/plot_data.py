import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('logs/data.csv', delimiter=',')

# Number of environments
num_envs = 3

# Separate data by environment and plot thrust values
for i in range(num_envs):
    data_env = data[i::num_envs]  # data for environment i

    # Assuming reset_buf is in column 0 and thrust values are in columns 2 to 5
    reset_buf = data_env[:, 0]
    reset_buf_d = data_env[:, 0]*5000
    thrusts = data_env[:, 2:6]

    # Assuming drone position is in columns 6 to 8 and hoop position is in columns 9 to 11
    drone_pos = data_env[:, 6:9]
    hoop_pos = data_env[:, 9:12]

    # Create a plot for each environment
    plt.figure()
    for j in range(thrusts.shape[1]):
        plt.plot(thrusts[:, j], label=f'Thrust {j+1}')
    plt.plot(reset_buf_d, label='Reset Buffer', linestyle='--')
    plt.title(f'Environment {i+1}')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()

    # Create a 3D plot for the drone and hoop positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Split drone_pos into subarrays each time reset_buf is 1, and plot each subarray separately
    splits = np.where(reset_buf == 1)[0]
    for start, end in zip(np.r_[0, splits+1], np.r_[splits+1, len(drone_pos)]):
        ax.plot(drone_pos[start:end, 0], drone_pos[start:end, 1], drone_pos[start:end, 2])

    # Draw a square frame around the hoop position for each subarray
    for hp in hoop_pos[splits]:
        x, y, z = hp
        for dy in [0.3]:
            ax.plot([x-0.35, x+0.35, x+0.35, x-0.35, x-0.35], [y+dy]*5, [z-0.35, z-0.35, z+0.35, z+0.35, z-0.35], color='b')

    ax.set_title(f'Environment {i+1} - 3D Positions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Set the same scale for all axes
    # Set the same scale for all axes
    max_range = np.array([drone_pos[:,0].max()-drone_pos[:,0].min(), drone_pos[:,1].max()-drone_pos[:,1].min(), drone_pos[:,2].max()-drone_pos[:,2].min()]).max() / 2.0
    mid_x = (drone_pos[:,0].max()+drone_pos[:,0].min()) * 0.5
    mid_y = (drone_pos[:,1].max()+drone_pos[:,1].min()) * 0.5
    mid_z = (drone_pos[:,2].max()+drone_pos[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


# Show the plots
plt.show()