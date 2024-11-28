import setup_path
import gym
from gym import register
import airgym
from airgym.envs.drone_env import AirSimDroneEnv
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

register(
    id='airsim-drone-sample-v0',
    entry_point='__main__:AirSimDroneEnv',
    kwargs={
        'ip_address': '127.0.0.1',
        'step_length': 0.25,
        'image_shape': (84, 84, 1),

        # "render_modes": "rgb_array",
    }
)
# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airsim-drone-sample-v0"
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=500000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="HongJun/output/model",
    log_path="HongJun/output/log",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {"callback": callbacks}
# kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=int(5e5),
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("dqn_airsim_drone_policy")
