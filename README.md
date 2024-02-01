# Isaac Gym Drone Env

### About this repository

This repository contains a drone environment created based on the vectorized environment provided in Isaac Gym Env Repository [Link](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)


### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 4
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```
### Tasks
* Drone: A basic drone designed to reach a target point or marker and attempt to stabilize itself there.
<video width="640" height="480" controls>
  <source src="assets/repo/Drone.mp4" type="video/mp4">
</video>


* DroneHoops: A drone designed to fly toward a hoop and pass through it.
<video width="640" height="480" controls>
  <source src="assets/repo/DroneHoops.mp4" type="video/mp4">
</video>

### Running the training

To train your first policy, navigate to the `isaacgymenvs` directory:

```bash
cd isaacgymenvs
```

Then run the following command:

```bash
python train.py task=DroneHoops
```

You can also train headlessly with the following command:

```bash
python train.py task=DroneHoops headless=True
```


### Loading trained models // Checkpoints

Checkpoints are saved in the folder `runs/EXPERIMENT_NAME/nn` where `EXPERIMENT_NAME` 
defaults to the task name, but can also be overridden via the `experiment` argument.

To load a trained checkpoint and continue training, use the `checkpoint` argument:

```bash
python train.py task=DroneHoops checkpoint=runs/DroneHoops/nn/DroneHoops.pth
```

To load a trained checkpoint and only perform inference (no training), pass `test=True` 
as an argument, along with the checkpoint name. To avoid rendering overhead, you may 
also want to run with fewer environments using `num_envs=64`:

```bash
python train.py task=DroneHoops checkpoint=runs/DroneHoops/nn/DroneHoops.pth test=True num_envs=64
```
