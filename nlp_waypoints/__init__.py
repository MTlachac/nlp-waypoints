from gym.envs.registration import register

register(
  id='nlp-waypoints-v0',
  entry_point='nlp_waypoints.envs:QuadrupedEnv'
)
