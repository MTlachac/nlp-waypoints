import math
import random
import numpy as np
import gym
from gym.utils import seeding
from gym import spaces
import pybullet as p
import pybullet_data
from pyquaternion import Quaternion

import sys
sys.path.append("../dataGeneration")
sys.path.append("../../dataGeneration")
from dataGeneration import generateTestData

###  Arguments (see trainRL.py or  runRL.py for how to set these from training/testing scripts)
# gui: True if Pybullet window should render (leave False for training)
# latency: how many times stepSimulation() in Pybullet is called per training step
# maxSpeed: maximum value for action that sets joint speeds
# useHyq: True to use HYQ quadrupedal robot, False to use much simpler tumbling robot
# continuousActions: if True, actions are mapped directly to joint velocities; if False, actions are
#                    selected from (-maxSpeed, 0, maxSpeed)
# requireAngle: factor in the desired angle into rewards and goal checking
# distanceThreshold: how close the robot has to be to the goal to be done (in meters)
# angleThreshold: if requireAngle, how close the robot's angle must be to the goal (in radians)
# maxSteps: how many steps the environment can run before quitting if the goal hasn't been reached
# seekOrigin: True if robot should start away from the origin and navigate to it
#             False if robot should navigate from the origin to a set goal (generated or passed in)
#             If false, robot also observes goal position/orientation
# resetPosition: True if robot should begin at the origin after every reset
#                False if the robot should only reset once it's run out of goals to reach (useful
#                for applying learned policy, maybe useful for training)
#                meaningless if seekOrigin is True, since the robot will be reset away from origin
###

class QuadrupedEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
                  gui = False,               
                  latency = 1,
                  maxSpeed = 5,
                  useHyq = True,
                  continuousActions = False,
                  requireAngle = False,
                  distanceThreshold = 0.1,
                  angleThreshold = 0.1745,  # 10 degrees
                  maxSteps = 50000,
                  seekOrigin = True,
                  resetPosition = True
                ):

        # set up pybullet
        self.physicsClient = p.connect(p.GUI) if gui else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

        # configurable variables (from kwargs)
        self.latency = latency
        self.maxSpeed = maxSpeed
        self.numJoints = 17 if useHyq else 2
        self.urdf = urdf = "../robotDescription/" + ("hyq" if useHyq else "tumblebot") + ".urdf"
        self.continuousActions = continuousActions
        self.requireAngle = requireAngle
        self.distanceThreshold = distanceThreshold
        self.angleThreshold = angleThreshold
        self.maxSteps = maxSteps
        self.seekOrigin = seekOrigin
        self.resetPosition = resetPosition

        self.goal = np.array([])
        self.goalIndex = 0

        # set up action and observation spaces
        if continuousActions:
          self.action_space = spaces.Box(np.array([-self.maxSpeed] * self.numJoints),
                                         np.array( [self.maxSpeed] * self.numJoints))
        else:
          self.action_space = spaces.MultiDiscrete([3] * self.numJoints)
        # 3 goal parameters if goal isn't origin, 3 body position, 4 body orientation, joint values
        v = 3 if seekOrigin else 6
        self.observation_space = spaces.Box(
                              np.array([-np.inf] * v + [-1] * 4 + [-2*np.pi] * self.numJoints),
                              np.array( [np.inf] * v +  [1] * 4 +  [2*np.pi] * self.numJoints))

        # extra data to pass to Tensorboard
        self.resetCount = 0
        self.currentDistance = 0
        self.currentAngle = 0
        self.currentReward = 0
        self.stepCount = 0


    def step(self, action):
        self.stepCount += 1
        self.completeAction(action)

        for i in range(self.latency):
            p.stepSimulation()

        observation = self.getObservation()
        if self.seekOrigin:
          position, orientation = observation[0:3], observation[3:7]
          goal = np.array([0.0, 0.0, 0.0])
        else:
          position, orientation = observation[3:6], observation[6:10]
          goal = self.goal[self.goalIndex]
        
        reward = self.getReward(position, orientation, goal)
        done = self.isDone(position, orientation, goal)

        return observation, reward, done, {}


    def reset(self):
        self.stepCount = 0
        self.resetCount += 1
    
        if self.seekOrigin:
            p.resetSimulation()
            p.setGravity(0,0,-9.81)
            planeId = p.loadURDF("plane.urdf")

            angle = random.uniform(0, 2 * math.pi)
            startPosition = [2 * math.cos(angle), 2 * math.sin(angle), 1]
            self.botId = p.loadURDF(self.urdf, startPosition)

        else:
            self.goalIndex += 1
            newGoal = False
            if self.goalIndex >= self.goal.shape[0]:
                self.goalIndex = 0
                _, _, newGoal = generateTestData()
                self.goal = self.convertToGlobal(newGoal)
                newGoal = True

            if self.resetPosition or newGoal:
                p.resetSimulation()
                p.setGravity(0,0,-9.81)
                planeId = p.loadURDF("plane.urdf")

                self.botId = p.loadURDF(self.urdf, [0, 0, 1])
 
        return self.getObservation()


    # pybullet handles the rendering, this avoids a NotImplementedError being thrown by gym
    def render(self, mode='human', close=False):
        return


    def completeAction(self, action):
        if self.continuousActions:
          velocities = action
        else:
          velocities = [(-self.maxSpeed, 0, self.maxSpeed)[a] for a in action]
        
        p.setJointMotorControlArray(bodyUniqueId = self.botId,
                                    jointIndices = range(self.numJoints),
                                    controlMode = p.VELOCITY_CONTROL,
                                    targetVelocities = velocities)


    def getObservation(self):
        position, orientation = p.getBasePositionAndOrientation(self.botId)
        jointStates = p.getJointStates(self.botId, range(self.numJoints))
        jointPositions = [joint[0] for joint in jointStates]

        if self.seekOrigin:
            return np.array(list(position) + list(orientation) + jointPositions)
        return np.array(list(self.goal[self.goalIndex]) + list(position) +
                        list(orientation) + jointPositions)


    def getReward(self, position, orientation, goal):
        dist = math.sqrt((position[0]- goal[0])**2 + (position[1]- goal[1])**2)
        angle = self.angleDifference(orientation, goal[2])
        reward = - dist - (angle if self.requireAngle else 0)

        # extra data for Tensorboard
        self.currentDistance = dist
        self.currentAngle = angle
        self.currentReward = reward

        return reward


    def isDone(self, position, orientation, goal):
        dist = math.sqrt((position[0]- goal[0])**2 + (position[1]- goal[1])**2)
        angle = self.angleDifference(orientation, goal[2])
        reachedAngle = (angle <= self.angleThreshold) if self.requireAngle else True
        reachedGoal = (dist <= self.distanceThreshold) and reachedAngle

        return reachedGoal or (self.stepCount >= self.maxSteps)


    # calculates angle difference between theta in plane and orientation from pybullet
    def angleDifference(self, orientation, theta):
        q = Quaternion(orientation[3], orientation[0], orientation[1], orientation[2]) 
        v1 = q.rotate(np.array([1., 0., 0.]))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        v2 = R @ np.array([1, 0, 0]).T
        return np.absolute(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


    def convertToGlobal(self, goal):
        newGoal = np.zeros(goal.shape)
        newGoal[0,:] = goal[0,:]
        for i in range(1, newGoal.shape[0]):
            c, s = np.cos(newGoal[i-1,2]), np.sin(newGoal[i-1,2])
            R = np.array([[c, -s], [s, c]])
            translation = R @ goal[i,:2]
            newGoal[i,:2] = newGoal[i-1,:2] + translation
            newGoal[i,2] = newGoal[i-1,2] + goal[i,2]

        return newGoal


    # used for applying model to specific goal, not for training
    def setGoal(self, goal, isGlobal = False):
      if not isGlobal:
        goal = self.convertToGlobal(goal)
      self.goal = goal


