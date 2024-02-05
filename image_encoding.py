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


# weights = torchvision.models.resnet.ResNet18_Weights.DEFAULT
weights = torchvision.models.resnet.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights)
model.eval()


def _forward_impl(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    logits = model.fc(x)
    latent = torch.log(1e-5 + x)
    return logits, latent[..., ::10]

model._forward_impl = _forward_impl


def encode(uint8_rgb_image):
    """ Compute a latent vector for a raw RGB image. """

    # the preproceesor expects `PIL.Image`s:
    rgb_image = Image.fromarray(uint8_rgb_image[:, :, ::-1])
    input_tensor = preprocess(rgb_image)
    with torch.no_grad():
        _, encoding = model(input_tensor.unsqueeze(0))
        return encoding.squeeze(0).numpy()


