import time
import pybullet as p
import pybullet_data

# joint ID to visualize (move and print info and state)
JOINT_ID = 3

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)

planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 2]
botId = p.loadURDF("hyq.urdf", startPos)

n = p.getNumJoints(botId)
print("Number of joints:", n)
print("Joint " + str(JOINT_ID) + ":")
print(p.getJointInfo(botId,3))

for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
    p.setJointMotorControl2(bodyUniqueId = botId,
                            jointIndex = JOINT_ID,
                            controlMode = p.VELOCITY_CONTROL,
                            targetVelocity = 10,
                            force = 1000)
    print(p.getJointState(botId, JOINT_ID))

p.disconnect()

