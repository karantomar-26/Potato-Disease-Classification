import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet18(pretrained=True)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model
