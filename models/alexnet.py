from torchvision.models import AlexNet
from torchvision import transforms
import torch

class AlexNet_with_trans(AlexNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = transforms.Resize([256,256])(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

