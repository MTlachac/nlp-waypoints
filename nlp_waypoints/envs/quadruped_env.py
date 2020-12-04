import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
from pyquaternion import Quaternion

import sys
sys.path.append("../dataGeneration")
sys.path.append("../../dataGeneration")
from dataGeneration import generateTestData

# the goal will be multiple points - sim ends when the last goal is reached
# loss is calculated wrt the current goal
# maybe divide loss by sentence length?

# set flag to true for visualizing
GUI = False

class QuadrupedEnv(gym.Env):
  metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
   
  def __init__(self):
    self.physicsClient = p.connect(p.GUI if GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    self.maxSpeed = 10
    self.numJoints = 17
    self.action_space = spaces.Box(np.array([-self.maxSpeed] * self.numJoints),
                                   np.array([self.maxSpeed] * self.numJoints))
    # 3 goal parameters, 3 body position, 4 body orientation, joint values
    self.observation_space = spaces.Box(
                              np.array([-np.inf] * 6 + [-1] * 4 + [-2*np.pi] * self.numJoints),
                              np.array([np.inf] * 6 + [1] * 4 + [2*np.pi] * self.numJoints))

    self.distanceThreshold = 0.1
    self.angleThreshold = 0.1745  # 10 degrees

  def step(self, action):
    p.setJointMotorControlArray(bodyUniqueId = self.botId,
                                jointIndices = range(self.numJoints),
                                controlMode = p.VELOCITY_CONTROL,
                                targetVelocities = action)
    p.stepSimulation()

    observation = self.getObservation()    
    reachedWaypoint = self.checkGoal(observation[2:5], observation[5:9])
    reward = self.getReward(observation[2:5], observation[5:9], reachedWaypoint) 

    return observation, reward, (self.goalIndex >= self.goal.shape[0]), {}

  def reset(self):
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    self.planeId = p.loadURDF("plane.urdf")
    try:
      self.botId = p.loadURDF("../../robotDescription/hyq.urdf", [0,0,1])
    except:
      self.botId = p.loadURDF("../robotDescription/hyq.urdf", [0,0,1]) 

    _, _, self.goal = generateTestData()
    self.convertGoalToGlobal()
    self.goalIndex = 0

    return self.getObservation()

  # pybullet handles the rendering
  def render(self, mode='human', close=False):
    return

  def getObservation(self):
    position, orientation = p.getBasePositionAndOrientation(self.botId)
    jointStates = p.getJointStates(self.botId, range(self.numJoints))
    jointPositions = [joint[0] for joint in jointStates]

    return np.array((list(self.goal[self.goalIndex,:]) + list(position) +
                     list(orientation) + jointPositions))

  # calculates angle difference between theta in plane and q
  # used for calculating reward and checking whether goal is reached
  def angleDifference(self, q, theta):
    v1 = q.rotate(np.array([1., 0., 0.]))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    v2 = R @ np.array([1, 0, 0]).T
    return np.absolute(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

  # return True if an intermediate goal has been reached
  def checkGoal(self, position, orientation):
    q = Quaternion(orientation[3], orientation[0], orientation[1], orientation[2])
    angleToGoal = self.angleDifference(q, self.goal[self.goalIndex, 2])
    distToGoal = np.linalg.norm(self.goal[self.goalIndex,:2] - position[2:])

    if (distToGoal < self.distanceThreshold) and (angleToGoal < self.angleThreshold):
      self.goalIndex += 1
      return True
    return False

  def getReward(self, position, orientation, reachedWaypoint): 
    q = Quaternion(orientation[3], orientation[0], orientation[1], orientation[2])
    angleToGoal = self.angleDifference(q, self.goal[self.goalIndex, 2])
    distToGoal = np.linalg.norm(self.goal[self.goalIndex,:2] - position[2:])

    reward = - (distToGoal + angleToGoal)
    if reachedWaypoint:
      reward += 100  # extra encouragement to reach a waypoint
    return reward

  # used for applying model to specific goal, not for training
  def setGoal(self, goal):
    self.goal = goal
    self.convertGoalToGlobal()

  def convertGoalToGlobal(self):
    newGoal = np.zeros(self.goal.shape)
    newGoal[0,:] = self.goal[0,:]
    for i in range(1, newGoal.shape[0]):
      c, s = np.cos(newGoal[i-1,2]), np.sin(newGoal[i-1,2])
      R = np.array([[c, -s], [s, c]])
      translation = R @ self.goal[i,:2]
      newGoal[i,:2] = newGoal[i-1,:2] + translation
      newGoal[i,2] = newGoal[i-1,2] + self.goal[i,2]

    self.goal = newGoal

    
