import gym

class QuadrupedEnv(gym.Env):
  metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
 
  
  def __init__(self):
    return

  def step(self, action):
    return

  def reset(self):
    return

  def render(self, mode='human', close=False):
    return

