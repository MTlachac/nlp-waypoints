import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead
from matplotlib import pyplot as plt

import sys
sys.path.append("../dataGeneration")
sys.path.append("../modelTraining")
from dataGeneration import generateTestData
from splittingNet import SplittingNet
from waypointNet import WaypointNet, listToTensor


# for debugging, avoids scientific notation and limits precision
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)


def loadModels(splittingModelPath, waypointModelPath):
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModelWithLMHead.from_pretrained("bert-base-uncased")

  splittingNet = SplittingNet(model.bert)
  splittingNet.load_state_dict(torch.load(splittingModelPath, map_location=torch.device('cpu')))
  splittingNet.eval()

  waypointNet = WaypointNet(model.bert)
  waypointNet.load_state_dict(torch.load(waypointModelPath, map_location=torch.device('cpu')))
  waypointNet.eval()

  return (splittingNet, waypointNet, tokenizer)


# helper to strip first n words from sentence
def removeWords(text, n):
  words = text.split(" ")
  if len(words) <= n:
    return (text, "")

  i = sum([len(words[j]) for j in range(n)]) + n
  return (text[:i], text[i:])


# helper to convert waypoints to global frame for plotting
def convertToGlobal(waypoints):
  output = np.zeros(waypoints.shape)
  output[0,:] = waypoints[0,:]
  for i in range(1, output.shape[0]):
    c, s = np.cos(output[i-1,2]), np.sin(output[i-1,2])
    R = np.array([[c, -s], [s, c]])
    translation = R @ waypoints[i,:2]
    output[i,:2] = output[i-1,:2] + translation
    output[i,2] = output[i-1,2] + waypoints[i,2]

  return output


def runModels(text, splittingNet, waypointNet, tokenizer):
  with torch.no_grad():
    # sentences to subsentences
    phrases = []
    while True:
      enc = tokenizer.batch_encode_plus([text])
      X = listToTensor(enc["input_ids"])
      attn = listToTensor(enc["attention_mask"])
      output = splittingNet((X, attn))
     
      n = torch.argmax(output).item()
      phrase, text = removeWords(text, n)
      phrases.append(phrase)

      if text == "":
        break

    # subsentences to waypoints
    enc = tokenizer.batch_encode_plus(phrases)
    X = listToTensor(enc["input_ids"])
    attn = listToTensor(enc["attention_mask"])
    
    output = waypointNet((X, attn))
    return output.numpy()


if __name__ == "__main__":
  splittingNet, waypointNet, tokenizer = loadModels(
    "../models/sentenceSplitting/splittingBERT.pth",
    "../models/waypointGeneration/waypointBERT.pth")
   
  fig, _ax = plt.subplots(nrows=3, ncols=2)
  ax = _ax.flatten()
  
  for i in range(6): 
    sentence, subsentences, waypoints = generateTestData(minLength = 3, maxLength = 5)
    predictedWaypoints = runModels(sentence, splittingNet, waypointNet, tokenizer)

    ax[i].plot([0] + waypoints[:,0], [0] +  waypoints[:,1], label = "Desired")
    ax[i].plot([0] + predictedWaypoints[:,0], [0] + predictedWaypoints[:,1], label = "Predicted")
    ax[i].legend()

  plt.show() 
  
