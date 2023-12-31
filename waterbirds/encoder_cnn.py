import torch.nn as nn
from torchvision.models.densenet import densenet121


IMG_ENCODE_SIZE = 1024


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = densenet121(weights='IMAGENET1K_V1')
        del self.net.classifier
        self.net.classifier = nn.Identity()

    def forward(self, x):
        return self.net(x)