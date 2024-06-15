import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Define layers here...
        )

    def forward(self, x):
        return self.model(x)
