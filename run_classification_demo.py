from matplotlib import pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed

from discriminator import BiGaussianDiscriminator
from image_encoding import encode
from collector import DataCollector


if __name__ == "__main__":

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

    plt.ion()
    collector = DataCollector(image_encoder=encode,
                              discriminator=BiGaussianDiscriminator())

    shell()