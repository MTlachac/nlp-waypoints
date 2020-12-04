import gym
import nlp_waypoints
import os

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpLstmPolicy

# set up path to save models at
SAVE_PATH = "../models/RL/"
os.makedirs(SAVE_PATH, exist_ok = True)

# create environment
env = gym.make('nlp-waypoints-v0')
vecEnv = DummyVecEnv([lambda: env])

# callback for saving models
nSteps = 0
def callback(_locals, _globals):
  global nSteps
  nSteps += 1

  if nSteps % 50000 == 0:
    model.save(SAVE_PATH + str(nSteps))

  return True

model = PPO2(MlpLstmPolicy, vecEnv, nminibatches = 1,
             verbose = 1, tensorboard_log = "/tmp/tensorboard")
model.learn(total_timesteps = 1000000000, callback = callback)
