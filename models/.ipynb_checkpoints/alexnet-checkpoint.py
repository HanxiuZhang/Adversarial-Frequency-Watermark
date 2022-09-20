from torchvision.models import AlexNet
from torchvision import transforms
import torch

class AlexNet_with_trans(AlexNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = transforms.Resize([2224,224])(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

from typing import Optional, Any
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models._utils import _ovewrite_named_param

def alexnet(*, weights: Optional[AlexNet_Weights] = None, progress: bool = True, **kwargs: Any) -> AlexNet:
    weights = AlexNet_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = AlexNet_with_trans(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model