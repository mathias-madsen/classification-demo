import os
import torch
import torchvision
from PIL import Image


preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])


class ResNet50Encoder:

    def __init__(self, downsampling_factor=10):

        self.downsampling_factor = downsampling_factor

        # weights = torchvision.models.resnet.ResNet18_Weights.DEFAULT
        weights = torchvision.models.resnet.ResNet50_Weights.DEFAULT
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.eval()

        def _forward_impl(x):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.model.fc(x)
            latent = torch.log(1e-5 + x)
            return logits, latent

        self.model._forward_impl = _forward_impl

    def __repr__(self):
        return ("ResNet50Encoder(downsampling_factor=%s)" %
                self.downsampling_factor)

    def __call__(self, uint8_rgb_image, return_logits=False):
        """ Compute a latent vector for a raw RGB image. """

        # the preproceesor expects `PIL.Image`s:
        rgb_image = Image.fromarray(uint8_rgb_image[:, :, ::-1])
        input_tensor = preprocess(rgb_image)
        with torch.no_grad():
            logits, encoding = self.model(input_tensor.unsqueeze(0))
            logits = logits.squeeze(0).numpy()
            encoding = encoding.squeeze(0).numpy()
            encoding = encoding[::self.downsampling_factor]
            if return_logits:
                return logits, encoding
            else:
                return encoding


def encode(uint8_rgb_image):
    """ Compute a latent vector for a raw RGB image. """

    # the preproceesor expects `PIL.Image`s:
    rgb_image = Image.fromarray(uint8_rgb_image[:, :, ::-1])
    input_tensor = preprocess(rgb_image)
    with torch.no_grad():
        _, encoding = model(input_tensor.unsqueeze(0))
        return encoding.squeeze(0).numpy()


