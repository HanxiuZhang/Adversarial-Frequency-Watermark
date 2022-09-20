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


from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models._utils import _ovewrite_named_param
def alexnet(*, weights: Optional[AlexNet_Weights] = None, progress: bool = True, **kwargs: Any) -> AlexNet:
    """AlexNet model architecture from `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`__.

    .. note::
        AlexNet was originally introduced in the `ImageNet Classification with
        Deep Convolutional Neural Networks
        <https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`__
        paper. Our implementation is based instead on the "One weird trick"
        paper above.

    Args:
        weights (:class:`~torchvision.models.AlexNet_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.AlexNet_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.AlexNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.AlexNet_Weights
        :members:
    """

    weights = AlexNet_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = AlexNet(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model