import numpy as np
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


names_file = "/Users/mathias/Downloads/imagenet_classes.txt"
with open(names_file, "r") as f:
    IMAGENET_CLASS_NAMES = [s.strip() for s in f.readlines()]


def encode(uint8_rgb_image):
    """ Compute a latent vector for a raw RGB image. """

    # the preproceesor expects `PIL.Image`s:
    rgb_image = Image.fromarray(uint8_rgb_image[:, :, ::-1])
    input_tensor = preprocess(rgb_image)
    with torch.no_grad():
        _, encoding = model(input_tensor.unsqueeze(0))
        return encoding.squeeze(0).numpy()


if __name__ == "__main__":

    from PIL import Image

    import cv2 as cv
    from matplotlib import pyplot as plt

    cam = cv.VideoCapture(0)
    
    result, bgr = cam.read()
    rgb = bgr[:, :, ::-1]

    input_image = Image.fromarray(rgb[:, ::-1, :])
    input_tensor = preprocess(input_image)
    with torch.no_grad():
        logits, encoding = model(input_tensor.unsqueeze(0))
        logits = logits.squeeze(0)
        encoding = encoding.squeeze(0)

    plt.ion()

    figure, (left, right) = plt.subplots(figsize=(12, 6),
                                         ncols=2,
                                         width_ratios=(5, 1))

    cam_display = left.imshow(bgr[:, ::-1, ::-1])
    left.axis("off")

    dot_plot, = right.plot([], [], ".")
    figure.suptitle("|||||", fontsize=48)
    right.set_xlim(-10, +10)
    right.set_ylim(-1, len(encoding) + 1)
    right.axis("off")

    figure.tight_layout()

    while plt.fignum_exists(figure.number):

        result, bgr = cam.read()
        rgb = bgr[:, :, ::-1]

        input_image = Image.fromarray(rgb[:, ::-1, :])
        input_tensor = preprocess(input_image)
        with torch.no_grad():
            logits, encoding = model(input_tensor.unsqueeze(0))
            logits = logits.squeeze(0)
            encoding = encoding.squeeze(0)
        winner = IMAGENET_CLASS_NAMES[np.argmax(logits)]
        print(winner)

        cam_display.set_data(bgr[:, ::-1, ::-1])
        dot_plot.set_data(encoding, range(len(encoding)))
        figure.suptitle(winner, fontsize=48)

        plt.pause(1 / 12)
