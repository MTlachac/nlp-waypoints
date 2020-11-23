import torch
from torch import nn

# custom head for BERT
# credit to Martin Zablocki: https://zablo.net/blog/post/custom-classifier-on-bert-model-guide-polemo2-sentiment-analysis/
class WaypointNet(nn.Module):
  def __init__(self, base_model, base_output_size=768, hidden_size = 100, dropout=0.05):
    super().__init__()
    self.base_model = base_model

    self.lstm = nn.LSTM(input_size=base_output_size, hidden_size=hidden_size,
                        num_layers=2, dropout=dropout)
    self.linear = nn.Linear(hidden_size, 3)

  def forward(self, input, *args):
    X, attention_mask = input
    hidden_states = self.base_model(X, attention_mask = attention_mask)[0]
    lstm_out = self.lstm(hidden_states.permute(1,0,2))[0].permute(1,0,2)
    return self.linear(lstm_out[:, -1, :])

# helper for padding 2D lists into tensors
def listToTensor(list):
  maxLen = max(len(l) for l in list)
  tensor = torch.tensor([l + (maxLen - len(l)) * [0] for l in list])
  return tensor

