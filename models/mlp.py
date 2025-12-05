import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_inputs, num_classes, hidden_sizes=[5, 10, 5]):
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        
        layers = []
        input_dim = num_inputs
        
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        feat = self.features(x)
        out = self.classifier(feat)
        return out
