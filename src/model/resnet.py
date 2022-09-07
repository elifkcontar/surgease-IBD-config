from typing import List

import torch.nn as nn
import torchvision
#from torchvision.models import ResNet18_Weights

from .head import DenseBlock


class ResNet18MultiHead(nn.Module):
    def __init__(self, num_classes: List[int], device="cpu"):
        super().__init__()

        #self.backbone = torchvision.models.resnet18(
        #    weights=ResNet18_Weights.DEFAULT
        #).to(device)
        #Elif:Issue about loading ResNet18_Weights
        self.backbone = torchvision.models.resnet18(
            pretrained=True
        ).to(device) 
        self.backbone.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

        # self.clf0 = nn.Sequential(nn.Linear(1000, num_classes[0])).to(device)
        self.clf0 = DenseBlock(
            input_dim=1000, out_dim=num_classes[0], dropout=0.1
        )
        # self.clf1 = nn.Sequential(nn.Linear(1000, num_classes[1])).to(device)
        self.clf1 = DenseBlock(
            input_dim=1000, out_dim=num_classes[1], dropout=0.1
        )
        # self.clf2 = nn.Sequential(nn.Linear(1000, num_classes[2])).to(device)
        self.clf2 = DenseBlock(
            input_dim=1000, out_dim=num_classes[2], dropout=0.1
        )

    def forward(self, input):
        z = self.backbone(input)
        return self.clf0(z), self.clf1(z), self.clf2(z)


class ResNet18SingleHead(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=True).to(device)
        self.backbone.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.clf = nn.Sequential(nn.Linear(1000, num_classes)).to(device)

    def forward(self, input):
        z = self.backbone(input)
        return self.clf(z)
