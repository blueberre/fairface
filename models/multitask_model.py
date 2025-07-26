import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class FairFaceMultiTaskModel(nn.Module):
    def __init__(self, num_race_classes=7):
        super(FairFaceMultiTaskModel, self).__init__()
        
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer

        # Final shared feature size
        in_features = base_model.fc.in_features

        # Two separate classification heads
        self.gender_head = nn.Linear(in_features, 2)
        self.race_head = nn.Linear(in_features, num_race_classes)

    def forward(self, x):
        features = self.backbone(x).squeeze()

        gender_logits = self.gender_head(features)
        race_logits = self.race_head(features)

        return gender_logits, race_logits
    
if __name__ == "__main__":
    import torch
    model = FairFaceMultiTaskModel(num_race_classes=7)
    dummy_input = torch.randn(4, 3, 224, 224)
    gender_logits, race_logits = model(dummy_input)
    print("Gender logits:", gender_logits.shape)
    print("Race logits:", race_logits.shape)