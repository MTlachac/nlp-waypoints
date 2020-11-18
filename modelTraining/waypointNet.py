import torch
from torch import nn

# custom head for BERT
# credit to Martin Zablocki: https://zablo.net/blog/post/custom-classifier-on-bert-model-guide-polemo2-sentiment-analysis/
class WaypointNet(nn.Module):
  def __init__(self, base_model, base_output_size=768, dropout=0.05):
    super().__init__()
    self.base_model = base_model

    self.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(base_output_size, 3)
    )

  def forward(self, input, *args):
    X, attention_mask = input
    hidden_states = self.base_model(X, attention_mask = attention_mask)[0]

    return self.classifier(torch.sum(hidden_states, 1))

# helper for padding 2D lists into tensors
def listToTensor(list):
  maxLen = max(len(l) for l in list)
  tensor = torch.tensor([l + (maxLen - len(l)) * [0] for l in list])
  return tensor

