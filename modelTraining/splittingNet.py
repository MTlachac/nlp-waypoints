import torch
from torch import nn

# custom head for BERT
# credit to Martin Zablocki: https://zablo.net/blog/post/custom-classifier-on-bert-model-guide-polemo2-sentiment-analysis/
class SplittingNet(nn.Module):
  def __init__(self, base_model, n_classes=30, base_model_output_size=768, hidden_size = 100, dropout=0.05):
    super().__init__()
    self.bert = base_model
    self.lstm = nn.LSTM(input_size=base_model_output_size, hidden_size=hidden_size,
                        num_layers=2, dropout=dropout)
    self.linear = nn.Linear(100, 70)
    self.activation = nn.PReLU()
    self.linear2 = nn.Linear(70, 30)
    self.activation2 = nn.PReLU()

    # the classifier ended up unused, but it was part of the network when it was
    # trained so it's part of the saved .pth model
    self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
            nn.PReLU()
    )
    for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()
    
  def forward(self, input, *args):
    X, attention_mask = input
    hidden_states = self.bert(X, token_type_ids=None, attention_mask = attention_mask)
    
    output = self.lstm(hidden_states.last_hidden_state.permute(1,0,2))[0].permute(1,0,2)
    output = self.linear(output[:,-1,:])
    output = self.activation(output)
    output = self.linear2(output)
    output = self.activation2(output)
    output = output/torch.norm(output)
    return output

