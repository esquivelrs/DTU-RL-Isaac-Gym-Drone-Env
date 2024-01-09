import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt, cos, sin

# initialize gym
gym = gymapi.acquire_gym()

print("%#############################TEST")

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# set up the env grid
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# set random seed
np.random.seed(17)

for i in range(num_envs):
    print("Creating env %d" % i)
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # create ring
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    pose.p = gymapi.Vec3(0, 1.5, 0)  # Adjust the position of the ring

    # Create vertices for the ring
    ring_radius_outer = 0.5
    ring_radius_inner = 0.4
    ring_height = 0.1
    segments = 32
    vertices = []
    indices = []

    for j in range(segments):
        theta = j * (2 * 3.14159 / segments)
        x_outer = ring_radius_outer * cos(theta)
        y_outer = ring_radius_outer * sin(theta)
        x_inner = ring_radius_inner * cos(theta)
        y_inner = ring_radius_inner * sin(theta)

        # Add vertices for the outer and inner circles of the ring
        vertices.extend([x_outer, y_outer, ring_height / 2.0])
        vertices.extend([x_inner, y_inner, ring_height / 2.0])

        # Add indices for the triangles
        indices.extend([2 * j, 2 * j + 1, (2 * j + 2) % (2 * segments)])
        indices.extend([(2 * j + 2) % (2 * segments), 2 * j + 1, (2 * j + 3) % (2 * segments)])

    # Create the ring mesh
    ring_mesh = gym.add_triangle_mesh(sim, np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), gymapi.TriangleMeshParams())

    # Create an actor for the ring
    ring_handle = gym.create_actor(env, ring_mesh, pose, None, 0, 0)
    gym.set_rigid_body_color(env, ring_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# create a local copy of the initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
