import gym
import nlp_waypoints
import os

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpLstmPolicy

LOAD_PATH = "../models/RL/500000.zip"

# create environment
env = gym.make('nlp-waypoints-v0', **{'gui': True})
vecEnv = DummyVecEnv([lambda: env])

# load model and run
model = PPO2.load(LOAD_PATH, env=vecEnv, policy=MlpLstmPolicy)
obs = vecEnv.reset()
state = None
done = [False]
while True:
  action, state = model.predict(obs, state=state, mask=done)
  obs, reward, done, _ = vecEnv.step(action)
  vecEnv.render()

