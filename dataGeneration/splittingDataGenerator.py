import numpy as np
import random
import math

# generates training data for a planar robot
# assumes x is forward, y is to the left wrt the robot
# assumes the robot should face the direction it most recently translated in

# word and value definitions
motionWords = [["go", "going"], 
               ["head", "heading"], 
               ["move", "moving"], 
               ["travel", "traveling"]]
rotationWords = [["turn", "turning"],
                 ["rotate", "rotating"]]
units = [["foot", "feet", 0.3048], 
         ["meter", "meters", 1], 
         ["inch", "inches", 0.0254], 
         ["yard", "yards", 0.9144], 
         ["centimeter", "centimeters", 0.01]]
fractions = [["a half", 0.5], 
             ["one half", 0.5], 
             ["a third", 1/3], 
             ["one third", 1/3], 
             ["two thirds", 2/3], 
             ["a fourth", 0.25], 
             ["one fourth", 0.25], 
             ["three fourths", 0.75]]
turns = [["left", np.array([0, 0, 1.0])],
         ["right", np.array([0, 0, -1.0])]]
directions = [[["forward", "up"], np.array([1, 0])],  
              [["backward", "back", "down"], np.array([-1, 0])], 
              [["left", "to the left"], np.array([0, 1])], 
              [["right", "to the right"], np.array([0, -1])]]
linkingWords = [["and", 0], 
                ["then", 0], 
                ["followed by", 1]]


def generateTestData(minLength = 1, maxLength = 3):
  length = random.randrange(maxLength - minLength + 1) + minLength
  sentence = ""
  motionIndex = 0
  phrases = [""] * length
  waypoints = np.zeros([length, 3])
  lastTurn = False
  for i in range(length):
    phrase = ""
    
    if (i!= 0):
        link = random.choice(linkingWords)
        phrase += link[0] + " "
        sentence += " "
        motionIndex = link[1]
      
    if ((not lastTurn) and random.random() < 0.5):  # rotation command
    
      rotationWord = random.choice(rotationWords)[motionIndex]
      phrase += rotationWord + " "
      direction = random.choice(turns)
      waypoint = np.copy(direction[1])
      
      if (random.random() < 0.5):  # turn a specific number of degrees
        angle = random.randrange(180) + 1
        phrase += str(angle) + " degrees "
        waypoint *= angle * math.pi / 180
      else:
        waypoint *= math.pi / 2
        
      if (random.random() < 0.5):
        phrase += "to the "
      phrase += direction[0]
      
      lastTurn = True
      
    else:  # translation command
    
      if (lastTurn or i==0 or random.random() < 0.5):
        phrase += random.choice(motionWords)[motionIndex] + " "
      
      distanceType = random.randrange(3)  # 0 = int, 1 = fraction, 2 = decimal
      if (distanceType < 3):  # integer/fraction
        dist = random.randrange(9) + 1  # don't want 0
      else:
        dist = 9.9 * random.random() + 0.1  # don't want 0
      phrase += ("%.4s" % dist) + " "
    
      if (distanceType == 1):  # fraction
        fraction = random.choice(fractions)
        phrase += "and " + fraction[0] + " "
        dist += fraction[1]  # now useful for calculating waypoint
    
      unit = random.choice(units)
      if (dist == 1):
        phrase += unit[0] + " "
      else:
        phrase += unit[1] + " "
       
      translation = np.array([0.0, 0.0])
      if (random.random() < 0.4):  # multiple directions
        firstDirectionIndex = random.randrange(len(directions))
        phrase += random.choice(directions[firstDirectionIndex][0]) + " and "
        translation += directions[firstDirectionIndex][1]
        
        directionIndex = random.randrange(len(directions))
        while ((directionIndex == firstDirectionIndex) or
               (directionIndex == 0 and firstDirectionIndex == 1) or
               (directionIndex == 1 and firstDirectionIndex == 0) or
               (directionIndex == 2 and firstDirectionIndex == 3) or
               (directionIndex == 3 and firstDirectionIndex == 2)):
          directionIndex = random.randrange(len(directions))
             
      else:
        directionIndex = random.randrange(len(directions))
        
      phrase += random.choice(directions[directionIndex][0])
      translation += directions[directionIndex][1]
      
      angle = np.arccos(translation[0] / np.linalg.norm(translation))
      angle *= (-1 if translation[1] < 0 else 1)
      translation *= dist * unit[2] / np.linalg.norm(translation)
      
      waypoint = np.concatenate((translation, np.array([angle])))
      lastTurn = False
      
    waypoints[i] = waypoint
    phrases[i] = phrase
    sentence += phrase
    expected = np.zeros((30))
    expected[len(phrases[0].split())] = 1
  
  return (sentence, expected)


# generates a batch of test data for subsentence to waypoint translation
# called generateTestData multiple times, so there's a mix of
# beginnings and middle sections of sentences
def generateTestBatch(batchSize):
  sentences = []
  splitarr = []
  #splits = np.zeros((batchSize, 30))
  
  i = 0
  while (i <= batchSize):
    sentence, split = generateTestData()
    sentences.append(sentence)
    splitarr.append(split)
    i += 1
  splits = np.vstack(splitarr)
  return (sentences, splits)
  
    
if __name__ == "__main__":
  sentence, split = generateTestData()
  print(sentence)
  print(split)
 
