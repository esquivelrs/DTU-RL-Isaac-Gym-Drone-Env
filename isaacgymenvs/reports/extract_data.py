import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Load the data
data = np.loadtxt('logs/data.csv', delimiter=',')

# Number of environments
num_envs = 5
plot = False

df = pd.DataFrame(data, columns=['reset_buf', 'progress_buf', 'thrust1', 'thrust2', 'thrust3', 'thrust4',
                                 'drone_x', 'drone_y', 'drone_z', 'hoop_x', 'hoop_y', 'hoop_z', 'reward', 'collision', 'target'])

# Add columns for the environment number and split number
df['env_num'] = np.repeat(np.arange(num_envs), len(df) // num_envs)
df['split_num'] = 0

# Update the split numbers
for i in range(num_envs):
    data_env = df[df['env_num'] == i]
    splits = np.where(data_env['reset_buf'] == 1)[0]
    for j, (start, end) in enumerate(zip(np.r_[0, splits+1], np.r_[splits+1, len(data_env)])):
        df.loc[data_env.index[start:end], 'split_num'] = j

print(df)
        
           


