import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from config import DEVICE

def get_resnet18(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改 fc 层
    return model.to(DEVICE)

if __name__ == "__main__":
    model = get_resnet18(num_classes=10)
    print(model)