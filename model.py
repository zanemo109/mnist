import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super(MLP, self).__init__()
        self.to_layer_1 = nn.Linear(input_size, hidden_1_size)
        self.to_layer_2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.to_output = nn.Linear(hidden_2_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.to_layer_1(x)
        x = self.relu(x)
        x = self.to_layer_2(x)
        x = self.relu(x)

        logits = self.to_output(x)
        return logits
    

