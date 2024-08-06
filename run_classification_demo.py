from matplotlib import pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed

from discriminator import BiGaussianDiscriminator
from image_encoding import encode as encode_with_resnet
from pretrained import encode as encode_with_pretrained
from collector import DataCollector


if __name__ == "__main__":

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

    plt.ion()
    collector = DataCollector(image_encoder=encode_with_pretrained,
                              discriminator=BiGaussianDiscriminator())

    shell()