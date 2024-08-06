from matplotlib import pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed

from discriminator import BiGaussianDiscriminator
from collector import DataCollector


if __name__ == "__main__":

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

    plt.ion()

    if True:
        from image_encoding import encode as encode_with_resnet
        collector = DataCollector(image_encoder=encode_with_resnet,
                                  discriminator=BiGaussianDiscriminator(205))
    else:
        from pretrained import encode as encode_with_pretrained
        collector = DataCollector(image_encoder=encode_with_pretrained,
                                  discriminator=BiGaussianDiscriminator(1707))

    shell()